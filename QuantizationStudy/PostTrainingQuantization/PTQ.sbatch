#!/bin/bash
#SBATCH --job-name=PostTrainingQuantization
#SBATCH --output=log/PostTrainingQuantization_output_%j.txt
#SBATCH --partition=elec.gpu-es02.q
#SBATCH --gres=gpu:1

module load cuDNN/8.7.0.84-CUDA-11.8.0

python3 PostTrainingQuantization.py
