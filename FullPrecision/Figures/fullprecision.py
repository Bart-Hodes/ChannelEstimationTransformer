import pickle
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "no-latex"])

if __name__ == "__main__":
    with open("../loss.pkl", "rb") as f:
        loss_list = pickle.load(f)

x_label = [i for i in range(1, len(loss_list) + 1)]

plt.figure(figsize=(10, 6))
plt.xlabel("Number of bits", fontsize=14)
plt.ylabel("Loss NMSE dB", fontsize=14)
plt.title("Full precision accuracy", fontsize=14)
plt.plot(x_label, 10 * np.log(loss_list), marker="o")
plt.savefig("fullprecision.png")
