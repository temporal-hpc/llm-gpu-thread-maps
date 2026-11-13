#!/bin/bash
#SBATCH --job-name=thread-map-gpu
#SBATCH --partition=AI
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=gpu_benchmark_%j.out
#SBATCH --error=gpu_benchmark_%j.err
#
# Compiles and runs the block-level benchmark and the power trace on one A100.
# Submit from the repository root:   sbatch gpu/gpu_benchmark.sbatch
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
srun --container-image=docker://nvidia/cuda:12.4.1-devel-ubuntu22.04 \
     --container-mounts="$REPO:$REPO" --container-workdir="$REPO/gpu" \
     bash -c 'set -e
       for src in gpu_benchmark power_trace carpet_benchmark; do
         nvcc -O3 -std=c++14 -o $src $src.cu -L/usr/local/cuda/lib64/stubs -lnvidia-ml -lpthread
       done
       ./gpu_benchmark
       ./power_trace
       ./carpet_benchmark'
