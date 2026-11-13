#!/bin/bash
#SBATCH --job-name=thread-map-inference
#SBATCH --partition=AI
#SBATCH --gpus=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=inference_%j.out
#SBATCH --error=inference_%j.err
#
# Runs the 1,188 inferences on 4x A100 (40 GB) with Ollama 0.12.6.
# Submit from the repository root after running inference/make_payloads.py:
#     sbatch inference/inference.sbatch
#
# The version is pinned: newer Ollama releases fail to load the MXFP4 blobs
# of gpt-oss pulled in 2025. Server and driver run in ONE step of the official
# image so that the server sees the four GPUs (qwen3:235b needs 142 GB).
# OLLAMA_MODELS must point to a directory where the 11 models were pulled
# beforehand; the job itself has no internet access.
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"

srun --ntasks=1 \
     --container-image=docker://ollama/ollama:0.12.6 \
     --container-mounts="$REPO:$REPO,$OLLAMA_MODELS:$OLLAMA_MODELS" \
     bash -c "export OLLAMA_MODELS=$OLLAMA_MODELS
              export OLLAMA_KEEP_ALIVE=60m
              ollama serve 2>&1 | grep -viE '^\[GIN\]' &
              sleep 10
              bash $REPO/inference/drive.sh"
