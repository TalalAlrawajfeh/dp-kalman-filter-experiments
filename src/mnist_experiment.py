import math
import os
import time
import warnings
from collections import namedtuple

import jax
import jax.numpy as jnp
import tensorflow as tf
from absl import app, flags, logging

from jaxdpopt.models import create_train_state

keras = tf.keras

from jaxdpopt.jax_mask_efficient import (
    compute_per_example_gradients_physical_batch,
    add_trees,
    clip_physical_batch,
    accumulate_physical_batch,
    CrossEntropyLoss, compute_per_example_look_ahead_gradients_physical_batch
)
from jaxdpopt.dp_accounting_utils import calculate_noise

USE_GPU = False

flags.DEFINE_float('clipping_norm', 0.1, 'Clipping norm for the per-sample gradients.')
flags.DEFINE_integer('eval_every_n_steps', 100, 'How often to run eval.')
flags.DEFINE_string('experiment_name', '', 'Experiment name')
flags.DEFINE_integer('num_steps', 10000, 'Number of training steps.')
flags.DEFINE_string('optimizer_name', 'disk', 'Name of the optimizer')
flags.DEFINE_integer('rnd_seed', None, 'Initial random seed, if not specified then OS source of entropy will be used.')
flags.DEFINE_integer('train_device_batch_size', 128, 'Per-device training batch size.')
flags.DEFINE_boolean('use_gpu', False, 'If true then use GPU')

FLAGS = flags.FLAGS


def process_physical_batch_factory(loss_fn, kappa=0.0, gamma=0.0):
    def process_physical_batch(t, params):
        (
            state,
            previous_params,
            accumulated_clipped_grads,
            logical_batch_X,
            logical_batch_y,
            masks,
            clipping_norm) = params
        # slice
        physical_batch_size = FLAGS.train_device_batch_size
        image_dimension = 32

        start_idx = t * physical_batch_size
        pb = jax.lax.dynamic_slice(
            logical_batch_X,
            (start_idx, 0, 0, 0, 0),
            (physical_batch_size, 1, image_dimension, image_dimension, 3),
        )

        yb = jax.lax.dynamic_slice(logical_batch_y, (start_idx,), (physical_batch_size,))
        mask = jax.lax.dynamic_slice(masks, (start_idx,), (physical_batch_size,))

        # compute grads and clip
        if FLAGS.optimizer_name == 'disk':
            per_example_gradients1 = compute_per_example_look_ahead_gradients_physical_batch(state,
                                                                                             previous_params,
                                                                                             pb,
                                                                                             yb,
                                                                                             loss_fn,
                                                                                             gamma)
            per_example_gradients2 = compute_per_example_gradients_physical_batch(state, pb, yb, loss_fn)
            alpha = (1 - kappa) / (kappa * gamma)
            per_example_gradients = jax.tree_util.tree_map(lambda g1, g2: alpha * g1 + (1 - alpha) * g2,
                                                           per_example_gradients1, per_example_gradients2)
        else:
            per_example_gradients = compute_per_example_gradients_physical_batch(state, pb, yb, loss_fn)

        clipped_grads_from_pb = clip_physical_batch(per_example_gradients, clipping_norm)
        sum_of_clipped_grads_from_pb = accumulate_physical_batch(clipped_grads_from_pb, mask)
        accumulated_clipped_grads = add_trees(accumulated_clipped_grads, sum_of_clipped_grads_from_pb)

        return (
            state,
            previous_params,
            accumulated_clipped_grads,
            logical_batch_X,
            logical_batch_y,
            masks,
            clipping_norm
        )

    return jax.jit(process_physical_batch)


def tree_inner_product(tree1, tree2):
    leaves1, _ = jax.tree_util.tree_flatten(tree1)
    leaves2, _ = jax.tree_util.tree_flatten(tree2)
    return sum(jnp.vdot(x, y) for x, y in zip(leaves1, leaves2))


def tree_norm(tree):
    return jnp.sqrt(tree_inner_product(tree, tree))  # ||x|| = sqrt(<x, x>)


def tree_angle(tree1, tree2):
    dot_product = tree_inner_product(tree1, tree2)
    norm1 = tree_norm(tree1)
    norm2 = tree_norm(tree2)
    cos_theta = dot_product / (norm1 * norm2)
    return jnp.arccos(jnp.clip(cos_theta, -1.0, 1.0))  # Clip for numerical stability


def main(argv):
    del argv
    print('JAX host: %d / %d' % (jax.process_index(), jax.process_count()))
    print('JAX devices:\n%s' % '\n'.join(str(d) for d in jax.devices()), flush=True)
    if FLAGS.rnd_seed is not None:
        rnd_seed = FLAGS.rnd_seed
    else:
        rnd_seed = int.from_bytes(os.urandom(8), 'big', signed=True)
    print('Initial random seed %d', rnd_seed)

    # Some assertions on the flags
    assert FLAGS.optimizer_name in ['disk', 'momentum'], f'Unknown optimizer name {FLAGS.optimizer_name}'
    if len(FLAGS.experiment_name) > 0:
        assert FLAGS.experiment_name in ['disk', 'momentum'], f'Unknown experiment name {FLAGS.experiment_name}'


    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    dataset_size = len(train_labels)
    train_images = train_images.reshape(-1, 32, 32, 3)
    test_images = test_images.reshape(-1, 32, 32, 3)
    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    optimizer_config = namedtuple("Config", ["learning_rate"])
    optimizer_config.learning_rate = 1e-3

    target_delta = 1e-6
    target_epsilon = 1.0
    logical_bs = FLAGS.train_device_batch_size
    num_classes = 10
    image_dimension = 32
    physical_batch_size = FLAGS.train_device_batch_size
    clipping_norm = FLAGS.clipping_norm
    accountant = 'pld'
    kappa = 0.7
    gamma = 0.2

    state = create_train_state(
        model_name="small",
        num_classes=num_classes,
        image_dimension=image_dimension,
        optimizer_config=optimizer_config,
    )
    loss_fn = CrossEntropyLoss(state=state, num_classes=num_classes, resizer_fn=lambda x: x)

    if dataset_size * target_delta > 1.0:
        warnings.warn("Your delta might be too high.")

    subsampling_ratio = 1 / math.ceil(dataset_size / logical_bs)

    noise_std = calculate_noise(
        sample_rate=subsampling_ratio,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        steps=FLAGS.num_steps,
        accountant='pld',
    )

    from jaxdpopt.dp_accounting_utils import compute_epsilon
    from jaxdpopt.jax_mask_efficient import (
        get_padded_logical_batch,
        model_evaluation,
        add_Gaussian_noise,
        poisson_sample_logical_batch_size,
        setup_physical_batches,
        update_model,
    )

    times = []
    logical_batch_sizes = []

    process_physical_batch_fn = process_physical_batch_factory(loss_fn, kappa, gamma)

    previous_params = jax.tree_util.tree_map(lambda x: x * 0, state.params)
    previous_noisy_grad = jax.tree_util.tree_map(lambda x: x * 0, state.params)

    adam_mom1, adam_mom2 = None, None
    adam_mom1noisy, adam_mom2noisy = None, None
    per_example_gradients1, per_example_gradients2 = None, None
    clean_grads = None
    if FLAGS.experiment_name == 'momentum':
        adam_mom1 = jax.tree_util.tree_map(lambda x: x * 0, state.params)
        adam_mom2 = jax.tree_util.tree_map(lambda x: x * 0, state.params)
        adam_mom1noisy = jax.tree_util.tree_map(lambda x: x * 0, state.params)
        adam_mom2noisy = jax.tree_util.tree_map(lambda x: x * 0, state.params)
        clean_grads = jax.tree_util.tree_map(lambda x: x * 0, state.params)
    elif FLAGS.experiment_name == 'disk':
        per_example_gradients1 = jax.tree_util.tree_map(lambda x: x * 0, state.params)
        per_example_gradients2 = jax.tree_util.tree_map(lambda x: x * 0, state.params)

    # Count and print the number of trainable parameters
    num_trainable_params = sum(param.size for param in jax.tree_util.tree_leaves(state.params))
    print(f"Number of trainable parameters: {num_trainable_params}")

    inner_product_test_batch_x = test_images[:128].reshape(-1, 1, 32, 32, 3)
    inner_product_test_batch_y = test_labels[:128]

    for t in range(FLAGS.num_steps):
        sampling_rng = jax.random.key(t + 1)
        batch_rng, binomial_rng, noise_rng = jax.random.split(sampling_rng, 3)

        #######
        # poisson subsample
        actual_batch_size = poisson_sample_logical_batch_size(
            binomial_rng=binomial_rng, dataset_size=dataset_size, q=subsampling_ratio
        )

        # determine padded_logical_bs so that there are full physical batches
        # and create appropriate masks to mask out unnessary elements later
        masks, n_physical_batches = setup_physical_batches(
            actual_logical_batch_size=actual_batch_size,
            physical_bs=physical_batch_size,
        )

        # get random padded logical batches that are slighly larger actual batch size
        padded_logical_batch_X, padded_logical_batch_y = get_padded_logical_batch(
            batch_rng=batch_rng,
            padded_logical_batch_size=len(masks),
            train_X=train_images,
            train_y=train_labels,
        )

        padded_logical_batch_X = padded_logical_batch_X.reshape(-1, 1, image_dimension, image_dimension, 3)

        # cast to GPU
        if FLAGS.use_gpu:
            padded_logical_batch_X = jax.device_put(padded_logical_batch_X, jax.devices("gpu")[0])
            padded_logical_batch_y = jax.device_put(padded_logical_batch_y, jax.devices("gpu")[0])
            masks = jax.device_put(masks, jax.devices("gpu")[0])

        ### gradient accumulation
        params = state.params

        accumulated_clipped_grads0 = jax.tree.map(lambda x: 0.0 * x, params)

        start = time.time()

        # Main loop
        _, _, accumulated_clipped_grads, *_ = jax.lax.fori_loop(
            0,
            n_physical_batches,
            process_physical_batch_fn,
            (
                state,
                previous_params,
                accumulated_clipped_grads0,
                padded_logical_batch_X,
                padded_logical_batch_y,
                masks,
                clipping_norm))

        if FLAGS.use_gpu:
            actual_batch_size = jax.device_put(actual_batch_size, jax.devices("gpu")[0])

        noisy_grad = add_Gaussian_noise(noise_rng, accumulated_clipped_grads, noise_std, clipping_norm)
        if FLAGS.optimizer_name == 'disk':
            noisy_grad = jax.tree_util.tree_map(
                lambda g1, g2: (1 - kappa) * g1 + kappa * g2, previous_noisy_grad, noisy_grad)

        noisy_grad = jax.tree_util.tree_map(lambda x: x / actual_batch_size, noisy_grad)
        accumulated_clipped_grads = jax.tree_util.tree_map(lambda x: x / actual_batch_size, accumulated_clipped_grads)

        # update
        state = jax.block_until_ready(update_model(state, noisy_grad))

        end = time.time()
        duration = end - start

        times.append(duration)
        logical_batch_sizes.append(actual_batch_size)

        if t % FLAGS.eval_every_n_steps == 0:
            acc_iter = model_evaluation(
                state, test_images, test_labels, batch_size=num_classes, orig_image_dimension=image_dimension,
                use_gpu=FLAGS.use_gpu
            )
            print(f"\n Throughput at iteration {t:8}: {actual_batch_size / duration:.2f} samp/s, accuracy at iteration {t:8}: {100*acc_iter:5.2f}%", flush=True)

            # Compute privacy guarantees
            epsilon, delta = compute_epsilon(
                noise_multiplier=noise_std,
                sample_rate=subsampling_ratio,
                steps=t + 1,
                target_delta=target_delta,
                accountant=accountant,
            )
            privacy_results = {"accountant": accountant, "epsilon": epsilon, "delta": delta}
            print(privacy_results, flush=True)

            if FLAGS.experiment_name == 'momentum':
                # Rerun main loop without clipping
                # Main loop
                _, _, clean_grads, *_ = jax.lax.fori_loop(
                    0,
                    n_physical_batches,
                    process_physical_batch_fn,
                    (
                        state,
                        previous_params,
                        accumulated_clipped_grads0,
                        padded_logical_batch_X,
                        padded_logical_batch_y,
                        masks,
                        1000000.0))
                clean_grads = jax.tree_util.tree_map(lambda x: x / actual_batch_size, clean_grads)

                count = sum(jnp.sum(2 * jnp.square(param) > var) for param, var in zip(jax.tree_util.tree_leaves(adam_mom1), jax.tree_util.tree_leaves(adam_mom2)))
                print(f"Number of parameters where SNR > 1.: {100 * count / num_trainable_params:.2f}%")
                count = sum(jnp.sum(2 * jnp.square(param) > .1 * var) for param, var in zip(jax.tree_util.tree_leaves(adam_mom1), jax.tree_util.tree_leaves(adam_mom2)))
                print(f"Number of parameters where SNR > .1: {100 * count / num_trainable_params:.2f}%")
                
                norm_adam1 = tree_norm(adam_mom1)
                adam1_rescaled = jax.tree_util.tree_map(lambda x: x / norm_adam1, adam_mom1)
                norm_adam1_noisy = tree_norm(adam_mom1noisy)
                adam1_rescaled_noisy = jax.tree_util.tree_map(lambda x: x / norm_adam1_noisy, adam_mom1noisy)

                pred2 = jax.tree_util.tree_map(lambda x, y: x * (2 * jnp.square(x) > 0.1 * y), adam1_rescaled, adam_mom2)
                pred2_noisy = jax.tree_util.tree_map(lambda x, y: x * (2 * jnp.square(x) > 0.1 * y), adam1_rescaled_noisy, adam_mom2noisy)

                pred3 = jax.tree_util.tree_map(lambda x, y: x * (2 * jnp.square(norm_adam1 * x) > 0.1 * y), adam1_rescaled, adam_mom2)
                pred3_noisy = jax.tree_util.tree_map(lambda x, y: x * (2 * jnp.square(norm_adam1_noisy * x) > 0.1 * y), adam1_rescaled_noisy, adam_mom2noisy)
                norm_adam_pred = tree_norm(pred2)

                grad_norm = tree_norm(clean_grads)
                diff_norm1 = tree_norm(jax.tree_util.tree_map(lambda x, y: x - y, clean_grads, adam1_rescaled))
                diff_norm1_noisy = tree_norm(jax.tree_util.tree_map(lambda x, y: x - y, clean_grads, adam1_rescaled_noisy))
                diff_norm2 = tree_norm(jax.tree_util.tree_map(lambda x, y: x - y, clean_grads, pred2))
                diff_norm2_noisy = tree_norm(jax.tree_util.tree_map(lambda x, y: x - y, clean_grads, pred2_noisy))
                diff_norm3 = tree_norm(jax.tree_util.tree_map(lambda x, y: x - y, clean_grads, pred3))
                diff_norm3_noisy = tree_norm(jax.tree_util.tree_map(lambda x, y: x - y, clean_grads, pred3_noisy))

                angle1 = tree_angle(clean_grads, adam_mom1)
                angle1_noisy = tree_angle(clean_grads, adam_mom1noisy)
                print(f"grad_norm: {grad_norm:8.3f}, diff_norm1: {diff_norm1:8.4f}, diff_norm2: {diff_norm2:8.4f}, diff_norm3: {diff_norm3:8.4f}, angle: {angle1:8.2f}, mom1 norm {norm_adam_pred:8.3f}", flush=True)
                print(f"grad_norm: {norm_adam_pred:8.3f}, diff_norm1: {diff_norm1_noisy:8.4f}, diff_norm2: {diff_norm2_noisy:8.4f}, diff_norm3: {diff_norm3_noisy:8.4f}, angle: {angle1_noisy:8.2f}, mom1 norm {norm_adam1_noisy:8.3f}", flush=True)
                # print(f"grad_norm: {grad_norm:8.2f}, shortcut/public: {diff_norm2:7.2f}/{diff_norm2_noisy:8.2f}, angle: {angle1:8.2f}", flush=True)

            elif FLAGS.experiment_name == 'disk':
                pg1 = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), per_example_gradients1)
                pg2 = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), per_example_gradients2)

                angle_between = tree_angle(pg1, pg2)
                angle1 = tree_angle(pg1, accumulated_clipped_grads)
                angle2 = tree_angle(pg2, accumulated_clipped_grads)

                print(f"Angle between: {angle_between:8.3f}, angle1: {angle1:8.3f}, angle2: {angle2:8.3f}", flush=True)


        previous_params = params
        previous_noisy_grad = noisy_grad

        if FLAGS.experiment_name == 'disk':
            # Run at the end of the for-loop for the next iteration
            per_example_gradients1 = compute_per_example_look_ahead_gradients_physical_batch(
                state, previous_params, inner_product_test_batch_x, inner_product_test_batch_y, loss_fn, gamma)
            per_example_gradients2 = compute_per_example_gradients_physical_batch(
                state, inner_product_test_batch_x, inner_product_test_batch_y, loss_fn)
        if FLAGS.experiment_name == 'momentum':

            beta1, beta2 = 0.7, 0.9

            adam_mom1 = jax.tree_util.tree_map(lambda x, y: beta1 * x + (1-beta1) * y, adam_mom1, accumulated_clipped_grads)
            adam_mom2 = jax.tree_util.tree_map(lambda x, y: beta2 * x + (1-beta2) * jnp.square(y), adam_mom2, accumulated_clipped_grads)
            adam_mom1noisy = jax.tree_util.tree_map(lambda x, y: beta1 * x + (1.-beta1) * y, adam_mom1noisy, noisy_grad)
            adam_mom2noisy = jax.tree_util.tree_map(lambda x, y: beta2 * x + (1.-beta2) * jnp.square(y), adam_mom2noisy, noisy_grad)


if __name__ == '__main__':
    logging.set_verbosity(logging.ERROR)
    jax.config.config_with_absl()
    app.run(main)
