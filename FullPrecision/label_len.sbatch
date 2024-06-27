#!/bin/bash
#SBATCH --job-name=HyperparameterDff
#SBATCH --output=Figures/dff.txt
#SBATCH --partition=elec.gpu-es02.q
#SBATCH --gres=gpu:1

module load cuDNN/8.7.0.84-CUDA-11.8.0

python3 QuantizationAwareTraining.py --label_len 5
python3 QuantizationAwareTraining.py --label_len 10
python3 QuantizationAwareTraining.py --label_len 15
python3 QuantizationAwareTraining.py --label_len 20
