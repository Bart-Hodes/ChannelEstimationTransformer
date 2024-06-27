import matplotlib.pyplot as plt
import pickle
import pandas as pd
import torch
import numpy as np


with open("../loss_list_QAT_nearest.pkl", "rb") as f:
    data = pickle.load(f)


mean_values = [10 * np.log10(sum(array) / len(array)) for array, value in data]
x_values = [value for array, value in data]
print(data)
print(x_values)

plt.figure()
plt.plot(x_values, mean_values)
plt.title("QAT with partial nearest quantization SNR 40dB")
plt.xlabel("Q1.X")
plt.ylabel("NMSE (dB)")
plt.savefig("loss_QAT_nearest.png")
