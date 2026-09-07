#!/bin/bash
unset LD_LIBRARY_PATH

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"
export JAX_DEFAULT_MATMUL_PRECISION=highest

algos=(algo3 dhpg_fix2 mico_2 algo_aug)
tasks=(HumanoidRun HumanoidStand HumanoidWalk)
seeds=(0 1 2)

mkdir -p run_logs

for algo in "${algos[@]}"; do
  for task in "${tasks[@]}"; do
    for seed in "${seeds[@]}"; do
      name="${algo}_${task}_seed${seed}"
      echo "[$(date +'%F %T')] starting $name"
      python -m "$algo" \
        --init-temperature 0.1 \
        --task "$task" \
        --device cuda \
        --seed "$seed" \
        --log-dir logs_fresh \
        > "run_logs/${name}.log" 2>&1 \
        || echo "[$(date +'%F %T')] FAILED: $name"
      echo "[$(date +'%F %T')] finished $name"
    done
  done
done