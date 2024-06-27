#!/bin/bash
#SBATCH --job-name=TimingInformer
#SBATCH --output=log/trainInformer_output_%j.txt
#SBATCH --partition=elec.gpu-es02.q
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

module load cuDNN/8.7.0.84-CUDA-11.8.0

python3 TrainInformer.py
