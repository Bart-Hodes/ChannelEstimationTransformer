import os
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as scio
from data import *
import argparse
from torch.utils.data import Dataset, DataLoader
import math
from models.model import Informer, InformerStack, LSTM, RNN, GRU, InformerStack_e2e
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Assuming you have extracted the necessary values from argparse into variables:
enc_in = 16
dec_in = 16
c_out = 16
seq_len = 25
label_len = 10
pred_len = 5
factor = 5
d_model = 64
n_heads = 8
e_layers = 4
d_layers = 3
d_ff = 64
dropout = 0.05
attn = "full"
embed = "fixed"
activation = "gelu"
output_attention = True
distil = True
device = "cuda"  # Example value, replace this with your device choice
data = "0"


# Create InformerStack instance using the extracted values
informer = InformerStack(
    enc_in,
    dec_in,
    c_out,
    seq_len,
    label_len,
    pred_len,
    factor,
    d_model,
    n_heads,
    e_layers,
    d_layers,
    d_ff,
    dropout,
    attn,
    embed,
    activation,
    output_attention,
    distil,
    device,
)


def LoadBatch(H):
    """
    H: T * M * Nr * Nt
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


with open("channel.pickle", "rb") as handle:
    channel = pickle.load(handle)
# transformer model
model_load = "models/checkpoint/checkpoint.pth"
state_dicts = torch.load(model_load, map_location=torch.device("cpu"))
informer.load_state_dict(state_dicts["state_dict"])
print("informer  has been loaded!")

informer.eval()

batch_size, M, Nr, Nt = channel.shape
print(channel.shape)

data = LoadBatch(channel[:, :25, :, :])
inp_net = data.to(device)

print(data.shape)
# print(data)
enc_inp = inp_net
dec_inp = torch.zeros_like(enc_inp[:, -pred_len:, :]).to(device)
dec_inp = torch.cat([enc_inp[:, seq_len - label_len : seq_len, :], dec_inp], dim=1)

# print(enc_in)
# print(dec_in)


# informer
if output_attention:
    outputs_informer = informer(enc_inp, dec_inp)[0]
else:
    outputs_informer = informer(enc_inp, dec_inp)
outputs_informer = outputs_informer.cpu().detach()
outputs_informer = real2complex(np.array(outputs_informer))

outputs_informer = outputs_informer.reshape(
    [batch_size, pred_len, Nr, Nt]
)  # shape = [64, 3, 4, 2]

print(data.shape[1])

x = np.array(list(range(channel.shape[1])))
print(x)
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(x[-pred_len:], outputs_informer[0, :, 0, i].real)
    plt.plot(x, channel[0, :, 1, i].real)
    # plt.plot(x,channel[0,:,1,i].imag)
plt.savefig("pred.png")
