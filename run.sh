unset LD_LIBRARY_PATH

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0

device=0
python -m dhpg --task HumanoidRun --device cuda --target_task HumanoidStand --seed 0 --log-dir logs
