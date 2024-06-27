#!/bin/bash
#SBATCH --job-name=PartialStochastic
#SBATCH --output=log/PartialStochastic_output_%j.txt
#SBATCH --partition=elec.gpu.q
#SBATCH --gres=gpu:1

module load cuDNN/8.7.0.84-CUDA-11.8.0
cd ..
python ParallelExecution.py --weigth_quant_setting partial --rounding stochastic 
