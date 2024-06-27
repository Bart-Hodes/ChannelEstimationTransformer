import matplotlib.pyplot as plt
import pickle
import pandas as pd
import torch
import numpy as np
import scienceplots
from matplotlib.font_manager import FontProperties

plt.style.use(["science", "no-latex"])
monospace_font = FontProperties(family="monospace")

bitwidth_LSQ = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
loss_list_LSQ = [
    0.3217,
    0.1566,
    0.1275,
    0.1197,
    0.1165,
    0.1118,
    0.1121,
    0.1107,
    0.1110,
    0.1107,
]
floatingpoint = 0.1091

#####################################################################################################
with open("../loss_list_full_proximal.pickle", "rb") as f:
    data = pickle.load(f)

mean_values = [10 * np.log10(np.array(arr).flatten()[4]) for arr, _ in data]

print(mean_values)

x_values = [value for _, value in data]

print(x_values)

plt.figure(figsize=(6.4, 4.8))
plt.plot(x_values, mean_values, label="QAT Nearest-round", marker="o")

#####################################################################################################

with open("../loss_list_full_stochastic.pickle", "rb") as f:
    data = pickle.load(f)

mean_values = [10 * np.log10(np.array(arr).flatten()[4]) for arr, _ in data]

print(mean_values)

x_values = [value for _, value in data]

print(x_values)

plt.plot(x_values, mean_values, label="QAT Stochastic-round", marker="o")


#####################################################################################################

with open("../loss_list_partial_proximal.pickle", "rb") as f:
    data = pickle.load(f)

mean_values = [10 * np.log10(np.array(arr).flatten()[4]) for arr, _ in data]

print(mean_values)

x_values = [value for _, value in data]

print(x_values)

plt.plot(x_values, mean_values, label="INQ Proximal", marker="o")


#####################################################################################################

with open("../loss_list_partial_distant.pickle", "rb") as f:
    data = pickle.load(f)

mean_values = [10 * np.log10(np.array(arr).flatten()[4]) for arr, _ in data]

print(mean_values)

x_values = [value for _, value in data]

print(x_values)

plt.plot(x_values, mean_values, label="INQ Distant", marker="o")


#####################################################################################################
with open("../loss_list_partial_stochastic.pickle", "rb") as f:
    data = pickle.load(f)
print([np.array(arr).flatten()[4] for arr, _ in data])
mean_values = [10 * np.log10(np.array(arr).flatten()[4]) for arr, _ in data]

print(mean_values)

x_values = [value for _, value in data]

print(x_values)

plt.plot(x_values, mean_values, label="INQ Stochastic", marker="o")


#######################################################################################
# plt.plot(
#     bitwidth_LSQ,
#     10 * np.log10(loss_list_LSQ),
#     label="Learned Stepsize Quantization",
#     marker="o",
# )


###########################################################################################################
# Add horizontal lines for floating point accuracies
plt.axhline(
    y=10 * np.log10(floatingpoint),
    color="black",
    linestyle="--",
    label=f"FP32 Accuracy",
)

plt.xlim(4, 10)
plt.legend(prop=monospace_font)
plt.title("Qantization Aware Training SNR 21dB")
plt.xlabel("Bitwidth")
plt.ylim(-10, 0)
plt.grid(True)
plt.ylabel("Loss NMSE (dB)")
plt.savefig("QAT_loss.png", dpi=300)
plt.close()

#####################################################################################################

# plt.figure(figsize=(6.4, 4.8))
# with open("../loss_list_partial_proximal.pickle", "rb") as f:
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


# with open("../loss_list_partial_distant.pickle", "rb") as f:
#     data = pickle.load(f)


# mean_values = [
#     10 * np.log10(sum(np.array(arr).flatten()) / len(np.array(arr).flatten()))
#     for arr, _ in data
# ]
# x_values = [value for _, value in data]
# print(data)
# print(x_values)

# plt.plot(x_values, mean_values, label="Distant", marker="o")


# with open("../loss_list_partial_stochastic.pickle", "rb") as f:
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
