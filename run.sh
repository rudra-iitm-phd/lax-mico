unset LD_LIBRARY_PATH

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0

device=0
python -m test_2 --task HumanoidStand --device cuda --target_task CheetahRun --seed 0
