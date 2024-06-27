import matplotlib.pyplot as plt
import numpy as np
import pickle
import scienceplots

plt.style.use(["science", "no-latex"])

plt.figure(figsize=(6.4, 4.8))

bitwidth = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
loss_list = [
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

if __name__ == "__main__":

    plt.plot(
        bitwidth,
        10 * np.log10(loss_list),
        marker="o",
        label="LSQ",
    )

    plt.axhline(
        y=10 * np.log10(floatingpoint),
        color="black",
        linestyle="--",
        label=f"FP32 Accuracy",
    )

    plt.xlabel("Number of bits", fontsize=14)
    plt.ylabel("Loss NMSE dB", fontsize=14)
    plt.legend()
    plt.title("Learned stepsize quantization", fontsize=14)
    plt.grid(True)
    plt.xlim(2, 11)
    plt.xticks(np.arange(2, 12, 1))
    plt.savefig("loss_lsq_total.png", dpi=300)
