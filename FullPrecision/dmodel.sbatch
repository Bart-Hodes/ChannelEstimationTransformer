#!/bin/bash
#SBATCH --job-name=HyperparameterDmodel
#SBATCH --output=Figures/dmodel.txt
#SBATCH --partition=elec.gpu-es02.q
#SBATCH --gres=gpu:1

module load cuDNN/8.7.0.84-CUDA-11.8.0

python3 QuantizationAwareTraining.py --d_model 64
python3 QuantizationAwareTraining.py --d_model 96
python3 QuantizationAwareTraining.py --d_model 128
