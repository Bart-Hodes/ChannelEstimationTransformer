#!/bin/bash
#SBATCH --job-name=HyperparameterDff
#SBATCH --output=Figures/dff.txt
#SBATCH --partition=elec.gpu-es02.q
#SBATCH --gres=gpu:1

module load cuDNN/8.7.0.84-CUDA-11.8.0

python3 QuantizationAwareTraining.py --d_ff 64
python3 QuantizationAwareTraining.py --d_ff 96
python3 QuantizationAwareTraining.py --d_ff 128
python3 QuantizationAwareTraining.py --d_ff 192
python3 QuantizationAwareTraining.py --d_ff 256
