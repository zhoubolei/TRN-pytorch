#!/usr/bin/env bash
#SBATCH -t 1-00:00:00
#SBATCH --gres gpu:1
#SBATCH -p small
#SBATCH -c 5
#SBATCH --mem=80G

set -ex
nvidia-smi
python main.py something RGB \
    --arch BNInception \
    --num-segments 3 \
    --consensus-type TRN \
    --batch-size 64 \
    --workers $(nproc)
