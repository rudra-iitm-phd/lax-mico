unset LD_LIBRARY_PATH

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0

device=0
python -m main_hetero_sa_rep_dim --task CheetahRun --device cuda --target_task CheetahRun --seed 0 --lr 0.0003
