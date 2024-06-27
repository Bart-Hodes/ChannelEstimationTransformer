#!/bin/bash
#SBATCH --job-name=HyperparameterSNR
#SBATCH --output=Figures/SNR.txt
#SBATCH --partition=elec.gpu-es02.q
#SBATCH --gres=gpu:1

module load cuDNN/8.7.0.84-CUDA-11.8.0

python3 QuantizationAwareTraining.py --SNR 12
python3 QuantizationAwareTraining.py --SNR 14
python3 QuantizationAwareTraining.py --SNR 16
python3 QuantizationAwareTraining.py --SNR 18
python3 QuantizationAwareTraining.py --SNR 20
