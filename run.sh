unset LD_LIBRARY_PATH

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0

device=0
python -m vanilla_sac_noise --task HumanoidStand --device cuda --target_task HumanoidStand --seed 0 --log-dir noise_logs
