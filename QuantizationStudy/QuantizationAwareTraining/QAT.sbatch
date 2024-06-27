#!/bin/bash
#SBATCH --job-name=QATIncremental
#SBATCH --output=log/QATIncremental_output_%j.txt
#SBATCH --partition=elec.gpu-es02.q
#SBATCH --gres=gpu:1

echo "Waiting for scripts to finish..."

echo "Loading modules..."
module load cuDNN/8.7.0.84-CUDA-11.8.0
echo "Modules loaded"

echo "Running script..."
python3 QuantizationAwareTraining.py --weigth_quant_setting partial --rounding proximal 
