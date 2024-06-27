#!/bin/bash
#SBATCH --job-name=QATIncremental
#SBATCH --output=log/QATIncremental_output_%j.txt
#SBATCH --partition=elec.gpu-es02.q
#SBATCH --gres=gpu:1

module load cuDNN/8.7.0.84-CUDA-11.8.0

# Call batch scripts in parallel
for script in full_proximal.sbatch full_stochastic.sbatch partial_proximal.sbatch partial_distant.sbatch partial_stochastic.sbatch; do
    sbatch "$script" &
done

# Wait for all scripts to finish
echo "Waiting for scripts to finish..."
wait
echo "All scripts have finished."
