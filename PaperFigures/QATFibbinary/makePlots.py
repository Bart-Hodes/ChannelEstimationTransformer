import matplotlib.pyplot as plt
import pickle
import pandas as pd
import torch
import numpy as np
import scienceplots

plt.style.use(["science", "no-latex"])
plt.figure(figsize=(6.4, 4.8))

#####################################################################################################
with open("loss_list_full_proximal.pickle", "rb") as f:
    data = pickle.load(f)

mean_values = [
    10 * np.log10(sum(np.array(arr).flatten()) / len(np.array(arr).flatten()))
    for arr, _ in data
]

print(mean_values)

x_values = [value for _, value in data]

print(x_values)

plt.plot(x_values, mean_values, label="Full Nearest", marker="o")

#####################################################################################################

with open("loss_list_full_stochastic.pickle", "rb") as f:
    data = pickle.load(f)

mean_values = [
    10 * np.log10(sum(np.array(arr).flatten()) / len(np.array(arr).flatten()))
    for arr, _ in data
]

print(mean_values)

x_values = [value for _, value in data]

print(x_values)

plt.plot(x_values, mean_values, label="Full stochastic", marker="o")


#####################################################################################################

with open("loss_list_partial_proximal.pickle", "rb") as f:
    data = pickle.load(f)

mean_values = [
    10 * np.log10(sum(np.array(arr).flatten()) / len(np.array(arr).flatten()))
    for arr, _ in data
]

print(mean_values)

x_values = [value for _, value in data]

print(x_values)

plt.plot(x_values, mean_values, label="Partial nearest first", marker="o")


#####################################################################################################

with open("loss_list_partial_distant.pickle", "rb") as f:
    data = pickle.load(f)

mean_values = [
    10 * np.log10(sum(np.array(arr).flatten()) / len(np.array(arr).flatten()))
    for arr, _ in data
]

print(mean_values)

x_values = [value for _, value in data]

print(x_values)

plt.plot(x_values, mean_values, label="Partial distant first", marker="o")


#####################################################################################################
with open("loss_list_partial_stochastic.pickle", "rb") as f:
    data = pickle.load(f)

mean_values = [
    10 * np.log10(sum(np.array(arr).flatten()) / len(np.array(arr).flatten()))
    for arr, _ in data
]

print(mean_values)

x_values = [value for _, value in data]

print(x_values)

plt.plot(x_values, mean_values, label="Partial Stochastic", marker="o")

plt.legend()
plt.title("Qantization Aware Training Strategies Fibbinary")
plt.xlabel("Bitwidth")
plt.grid(True)
plt.ylabel("Loss NMSE (dB)")

plt.savefig("QAT_fibbinary.png")
plt.close()

#####################################################################################################

# plt.figure(figsize=(6.4, 4.8))
# with open("loss_list_partial_proximal.pickle", "rb") as f:
#     data = pickle.load(f)

# print(data)

# mean_values = [
#     10 * np.log10(sum(np.array(arr).flatten()) / len(np.array(arr).flatten()))
#     for arr, _ in data
# ]
# x_values = [value for _, value in data]
# print(data)
# print(x_values)

# plt.plot(x_values, mean_values, label="Nearest", marker="o")


# with open("loss_list_partial_distant.pickle", "rb") as f:
#     data = pickle.load(f)


# mean_values = [
#     10 * np.log10(sum(np.array(arr).flatten()) / len(np.array(arr).flatten()))
#     for arr, _ in data
# ]
# x_values = [value for _, value in data]
# print(data)
# print(x_values)

# plt.plot(x_values, mean_values, label="Distant", marker="o")


# with open("loss_list_partial_stochastic.pickle", "rb") as f:
#     data = pickle.load(f)
# print([sum(np.array(arr).flatten()) / len(np.array(arr).flatten()) for arr, _ in data])
# mean_values = [
#     10 * np.log10(sum(np.array(arr).flatten()) / len(np.array(arr).flatten()))
#     for arr, _ in data
# ]
# x_values = [value for _, value in data]
# print(data)
# print(x_values)

# plt.plot(x_values, mean_values, label="Random", marker="o")


# plt.legend()
# plt.title("QAT Incremental")
# plt.xlabel("Q1.X")
# plt.grid(True)
# plt.ylabel("Loss NMSE (dB)")

# plt.savefig("QAT_loss_incremental.png")
# plt.close()
