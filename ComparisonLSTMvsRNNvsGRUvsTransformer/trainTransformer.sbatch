#!/bin/bash
#SBATCH --job-name=trainTransformer
#SBATCH --output=log/trainTransformer_output_%j.txt
#SBATCH --partition=elec.gpu-es02.q
#SBATCH --gres=gpu:1

module load cuDNN/8.7.0.84-CUDA-11.8.0

python3 trainingScripts/TrainTransformer.py
