#!/bin/bash
# sbatch -p bml -A bml -w protagoras train.sh {model}
#SBATCH -c 64
#SBATCH -o ./out/%j.txt
#SBATCH -e ./err/%j.txt
model=$1
echo "Training model: $model"
python3 training.py --model $model