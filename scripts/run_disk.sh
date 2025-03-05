export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=1.0

datestring=`date +"%Y%m%d-%H%M%S"`
# python3 -m debugpy --listen 5678 --wait-for-client src/mnist_experiment.py \
python3 src/mnist_experiment.py \
    --use_gpu \
    --clipping_norm=1.0 \
    --train_device_batch_size=1024 \
    --experiment_name=disk \
    --optimizer_name=disk | tee log/disk_${datestring}_$RANDOM.txt
