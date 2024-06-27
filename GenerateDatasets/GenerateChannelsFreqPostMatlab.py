import numpy as np
import pickle
import torch
import scipy.io as scio
import os

# Load data
data_dict = scio.loadmat("Temp/RF_Channel.mat")
Channel = np.transpose(data_dict["H_channel"], axes=[0, 1, 2, 4, 3])

# Reshape the channel matrix
Batchsize, subcarrier, NumOfSymbols, Nr, Nt = Channel.shape
print(Channel.shape)
Channel = Channel.reshape([Batchsize * subcarrier, NumOfSymbols, Nr, Nt])


# Check if the folder 'Datasets' exists
if not os.path.exists("Datasets"):
    # Create the 'Datasets' folder
    os.makedirs("Datasets")

# Define file paths
fileTraining = f"Datasets/Seq_Len_{NumOfSymbols}_Beamforming2_CDLB.pickle"
fileValidation = f"Datasets/Seq_Len_{NumOfSymbols}_Beamforming2_CDLB__validate.pickle"

# Load existing training data if exists
try:
    with open(fileTraining, "rb") as handle:
        existing_data_train = pickle.load(handle)
except FileNotFoundError:
    existing_data_train = torch.empty(0)

# Load existing validation data if exists
try:
    with open(fileValidation, "rb") as handle:
        existing_data_validate = pickle.load(handle)
except FileNotFoundError:
    existing_data_validate = torch.empty(0)


trainValSplit = 0.8
trainSize = int(trainValSplit * Channel.shape[0])
ChannelTrain = Channel[:trainSize]
ChannelValidate = Channel[trainSize:]


# Concatenate new data to existing data
existing_data_train = torch.cat(
    (existing_data_train, torch.from_numpy(ChannelTrain)), dim=0
)
existing_data_validate = torch.cat(
    (existing_data_validate, torch.from_numpy(ChannelValidate)), dim=0
)


# Save updated training data
print(f"Writing training data with shape {existing_data_train.shape}")
with open(fileTraining, "wb") as handle:
    pickle.dump(existing_data_train, handle)
print("Training data updated.")

# Save updated validation data
print(f"Writing validation data with shape {existing_data_validate.shape}")
with open(fileValidation, "wb") as handle:
    pickle.dump(existing_data_validate, handle)
print("Validation data updated.")
