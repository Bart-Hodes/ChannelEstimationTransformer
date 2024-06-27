import torch
from model import StreamingTransformer
import pickle
import numpy as np


def LoadBatch(H):
    """
    LoadBatch function takes in a 4-dimensional array H and performs the following operations:
    1. Reshapes H to have dimensions [M, T, Nr * Nt], where M, T, Nr, Nt are the dimensions of H.
    2. Creates an array H_real of shape [M, T, Nr * Nt, 2] and initializes it with zeros.
    3. Assigns the real and imaginary parts of H to the corresponding dimensions of H_real.
    4. Reshapes H_real to have dimensions [M, T, Nr * Nt * 2].
    5. Converts H_real to a torch tensor of dtype torch.float32.
    6. Returns the resulting tensor H_real.

    Parameters:
    - H: A 4-dimensional numpy array of shape [T, M, Nr, Nt], where T, M, Nr, Nt are the dimensions of H.

    Returns:
    - H_real: A torch tensor of shape [M, T, Nr * Nt * 2] containing the real and imaginary parts of H.

    """
    M, T, Nr, Nt = H.shape
    H = H.reshape([M, T, Nr * Nt])
    H_real = np.zeros([M, T, Nr * Nt, 2])
    H_real[:, :, :, 0] = H.real
    H_real[:, :, :, 1] = H.imag
    H_real = H_real.reshape([M, T, Nr * Nt * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


# Create an instance of your transformer model
model = StreamingTransformer(16, 5, 25, 64, 8, 512, 6, 0.1)

# Set the model to evaluation mode
model.eval()

# Prepare your input data


# Load the dataset using pickle
with open("Datasets/Seq_Len_40_Beamforming_CDLB.pickle", "rb") as file:
    dataset = pickle.load(file)

# Your input data goes here
input_data = dataset

input_data = LoadBatch(input_data)

# Convert the input data to tensors
input_tensor = torch.tensor(input_data)


for i in range(input_tensor.shape[1] // 5):
    input = input_tensor[:, i * 5 : (i + 1) * 5, :]

    # Pass the input tensor through the model
    output = model(input)

# Print the output
