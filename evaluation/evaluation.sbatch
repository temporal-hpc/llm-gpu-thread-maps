#!/bin/bash
#SBATCH --job-name=thread-map-eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=evaluation_%j.out
#SBATCH --error=evaluation_%j.err
#
# Grades the 1,188 responses and rebuilds the CSV files of data/. CPU only.
# Submit from the repository root:   sbatch evaluation/evaluation.sbatch
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
srun --container-image=docker://python:3.11-slim \
     --container-mounts="$REPO:$REPO,$(readlink -f "$REPO/domains"):$(readlink -f "$REPO/domains")" \
     --container-workdir="$REPO" \
     bash evaluation/grade_all.sh "${SLURM_CPUS_PER_TASK:-32}"
