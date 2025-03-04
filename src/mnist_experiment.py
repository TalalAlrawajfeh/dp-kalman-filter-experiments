import math
import time
import warnings

from jaxdpopt.models import create_train_state
from collections import namedtuple

import tensorflow as tf
import jax

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

OPTIMIZER = 'disk'


def process_physical_batch_factory(loss_fn, kappa=0.0, gamma=0.0):
    def process_physical_batch(t, params):
        (
            state,
            previous_params,
            accumulated_clipped_grads,
            logical_batch_X,
            logical_batch_y,
            masks,
        ) = params
        # slice
        physical_batch_size = 128
        image_dimension = 28
        clipping_norm = 1

        start_idx = t * physical_batch_size
        pb = jax.lax.dynamic_slice(
            logical_batch_X,
            (start_idx, 0, 0, 0, 0),
            (physical_batch_size, 1, image_dimension, image_dimension, 1),
        )
        yb = jax.lax.dynamic_slice(logical_batch_y, (start_idx,), (physical_batch_size,))
        mask = jax.lax.dynamic_slice(masks, (start_idx,), (physical_batch_size,))

        # compute grads and clip
        if OPTIMIZER == 'disk':
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
        )

    return jax.jit(process_physical_batch)


def main():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    dataset_size = len(train_labels)

    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    optimizer_config = namedtuple("Config", ["learning_rate"])
    optimizer_config.learning_rate = 1e-3

    target_delta = 1e-6
    target_epsilon = 1.0
    num_steps = 1000
    logical_bs = 128
    num_classes = 10
    image_dimension = 28
    physical_batch_size = 128
    clipping_norm = 1.0
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
        steps=num_steps,
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

    for t in range(num_steps):
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

        padded_logical_batch_X = padded_logical_batch_X.reshape(-1, 1, image_dimension, image_dimension, 1)

        # cast to GPU
        if USE_GPU:
            padded_logical_batch_X = jax.device_put(padded_logical_batch_X, jax.devices("gpu")[0])
            padded_logical_batch_y = jax.device_put(padded_logical_batch_y, jax.devices("gpu")[0])
            masks = jax.device_put(masks, jax.devices("gpu")[0])

        print("##### Starting gradient accumulation #####", flush=True)
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
            )
        )
        noisy_grad = add_Gaussian_noise(noise_rng, accumulated_clipped_grads, noise_std, clipping_norm)
        if OPTIMIZER == 'disk':
            noisy_grad = jax.tree_util.tree_map(lambda g1, g2: (1 - kappa) * g1 + kappa * g2, previous_noisy_grad,
                                                noisy_grad)
        # update
        state = jax.block_until_ready(update_model(state, noisy_grad))

        end = time.time()
        duration = end - start

        times.append(duration)
        logical_batch_sizes.append(actual_batch_size)

        print(f"throughput at iteration {t}: {actual_batch_size / duration}", flush=True)

        acc_iter = model_evaluation(
            state, test_images, test_labels, batch_size=num_classes, orig_image_dimension=image_dimension,
            use_gpu=USE_GPU
        )
        print(f"accuracy at iteration {t}: {acc_iter}", flush=True)

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

        previous_params = params
        previous_noisy_grad = noisy_grad


if __name__ == '__main__':
    main()
