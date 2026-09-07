unset LD_LIBRARY_PATH

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"
export JAX_DEFAULT_MATMUL_PRECISION=highest

device=0
python -m algo4 --init-temperature 0.1 --task HopperHop --device cuda  --seed 0 --log-dir logs_new 
