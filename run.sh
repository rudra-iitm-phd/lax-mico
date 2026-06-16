unset LD_LIBRARY_PATH

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0

device=0
python -m main_changes --task CheetahRun --device cuda --target_task CheetahRun
