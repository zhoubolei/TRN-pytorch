#!/usr/bin/env bash
#SBATCH -t 1-10:00:00
#SBATCH --gres gpu:1
#SBATCH -p small
#SBATCH -c 5
set -ex

NPROC=$(nproc)
if [[ $NPROC -ge 10 ]]; then
    WORKERS=$(($NPROC / 2))
else
    WORKERS=$NPROC
fi

nvidia-smi
env | sort
python main.py jester RGB \
    --workers $WORKERS \
    $ARGS \
    "$@"