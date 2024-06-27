# =======================================================================================================================
# =======================================================================================================================
import os
import math
import time
import numpy as np
import scipy.io as scio
import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

pi = np.pi


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


def real2complex(data):
    B, P, N = data.shape
    data2 = data.reshape([B, P, N // 2, 2])
    data2 = data2[:, :, :, 0] + 1j * data2[:, :, :, 1]
    return data2


def noise(H, SNR):
    """
    Add complex white Gaussian noise to a given channel matrix.

    Args:
        H (torch.Tensor): The channel matrix.
        SNR (float): The signal-to-noise ratio in decibels.

    Returns:
        torch.Tensor: The channel matrix with added noise.
    """

    sigma = 10 ** (-SNR / 10)
    # Generate complex Gaussian noise with PyTorch
    real_part = torch.randn(*H.shape)
    imag_part = torch.randn(*H.shape)
    noise = np.sqrt(sigma / 2) * (real_part + 1j * imag_part)
    # Normalize the noise
    noise = noise * torch.sqrt(torch.mean(torch.abs(H) ** 2))

    return H + noise


def channelnorm(H):
    """
    Normalize the channel matrix H.

    Parameters:
    - H (torch.Tensor): The channel matrix.

    Returns:
    - H_normalized (torch.Tensor): The normalized channel matrix.
    """
    H_normalized = H / torch.sqrt(torch.mean(np.abs(H) ** 2))
    return H_normalized


class SeqData(Dataset):
    """
    Dataset class for sequence data.

    Args:
        dataset_name (str): The name of the dataset file.
        seq_len (int): The length of the input sequence.
        pred_len (int): The length of the prediction sequence.
        SNR (float, optional): The signal-to-noise ratio. Defaults to 20.
    """

    def __init__(self, dataset_name, seq_len, pred_len, sliding_window=0, SNR=20):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.length = seq_len + pred_len + sliding_window
        self.SNR = SNR
        self.sliding_window = sliding_window

        if dataset_name.endswith(".pickle"):
            with open(dataset_name, "rb") as handle:
                self.dataset = pickle.load(handle)
        if dataset_name.endswith(".mat"):
            dataset = scio.loadmat(dataset_name)
            self.dataset = dataset["H_channel"]

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the input sequence, noisy input sequence, input sequence for prediction, and predicted sequence.
        """
        H = self.dataset[idx]
        # [Subcarrier,SRSlot,Rx,RF-Chain]

        OFDMSlots, NRx, NTx = H.shape
        L = self.length

        start = np.random.randint(0, OFDMSlots - L + 1)
        end = start + L - self.sliding_window

        H = channelnorm(H)

        H_noise = noise(H, self.SNR)

        H = H[start : end + self.sliding_window, ...]
        H_noise = H_noise[start : end + self.sliding_window, ...]
        H_pred = H[self.seq_len : self.seq_len + self.pred_len, ...]
        H_seq = H_noise[0 : self.seq_len, ...]

        # print(H.shape, H_noise.shape, H_seq.shape, H_pred.shape)

        return H, H_noise, H_seq, H_pred
