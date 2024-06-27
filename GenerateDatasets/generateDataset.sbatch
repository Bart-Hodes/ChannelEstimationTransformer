#!/bin/bash
#SBATCH --job-name=GenerateChannelsMatlab
#SBATCH --output=log/GenerateChannelsMatlab_output_%j.txt
#SBATCH --partition=elec.gpu-es02.q
#SBATCH --gres=gpu:1


module load cuda11.5/toolkit/11.5.1 cuda11.5/profiler/11.5.1 cuda11.5/nsight/11.5.1 cuda11.5/fft/11.5.1 cuda11.5/blas/11.5.1 cudnn/8.1.1
module load matlab

python3 GenerateChannelsFreqPreMatlab.py || exit 666
matlab -nodisplay -r "run('./Beamforming2.m') ; exit"
python3 GenerateChannelsFreqPostMatlab.py

python3 GenerateChannelsFreqPreMatlab.py
matlab -nodisplay -r "run('./Beamforming2.m') ; exit"
python3 GenerateChannelsFreqPostMatlab.py

python3 GenerateChannelsFreqPreMatlab.py
matlab -nodisplay -r "run('./Beamforming2.m') ; exit"
python3 GenerateChannelsFreqPostMatlab.py

python3 GenerateChannelsFreqPreMatlab.py
matlab -nodisplay -r "run('./Beamforming2.m') ; exit"
python3 GenerateChannelsFreqPostMatlab.py

python3 GenerateChannelsFreqPreMatlab.py
matlab -nodisplay -r "run('./Beamforming2.m') ; exit"
python3 GenerateChannelsFreqPostMatlab.py

python3 GenerateChannelsFreqPreMatlab.py
matlab -nodisplay -r "run('./Beamforming2.m') ; exit"
python3 GenerateChannelsFreqPostMatlab.py

python3 GenerateChannelsFreqPreMatlab.py
matlab -nodisplay -r "run('./Beamforming2.m') ; exit"
python3 GenerateChannelsFreqPostMatlab.py

python3 GenerateChannelsFreqPreMatlab.py
matlab -nodisplay -r "run('./Beamforming2.m') ; exit"
python3 GenerateChannelsFreqPostMatlab.py

python3 GenerateChannelsFreqPreMatlab.py
matlab -nodisplay -r "run('./Beamforming2.m') ; exit"
python3 GenerateChannelsFreqPostMatlab.py

python3 GenerateChannelsFreqPreMatlab.py
matlab -nodisplay -r "run('./Beamforming2.m') ; exit"
python3 GenerateChannelsFreqPostMatlab.py

