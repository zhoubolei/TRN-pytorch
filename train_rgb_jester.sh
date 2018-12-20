#!/usr/bin/env bash
#SBATCH -t 1-00:00:00
#SBATCH --gres gpu:1
#SBATCH -p small
#SBATCH -c 5
set -ex


nvidia-smi
env | sort
python main.py jester RGB \
    -c configs/jester-v1.ini \
    --workers $(($(nproc) / 2)) \
    "$@"