#!/bin/bash
#SBATCH --job-name=TrainGRU
#SBATCH --output=log/TrainGRU_output_%j.txt
#SBATCH --partition=elec.gpu-es02.q
#SBATCH --gres=gpu:1

module load cuDNN/8.7.0.84-CUDA-11.8.0

python3 trainingScripts/TrainGRU.py
python3 trainingScripts/TrainRNN.py
python3 trainingScripts/TrainLSTM.py
python3 trainingScripts/TrainInformer.py
python3 trainingScripts/TrainTransformer.py
