import matplotlib.pyplot as plt
import numpy as np
import pickle
import scienceplots

plt.style.use(["science", "no-latex"])

plt.figure(figsize=(6.4, 4.8))


floatingpoint = 0.1091
if __name__ == "__main__":
    # Load the loss_list from the pickle file
    with open("../loss_list.pickle", "rb") as f:
        loss_list = pickle.load(f)

    # split over prediction length
    mean_values = [np.array(arr).flatten()[4] for arr, _ in loss_list]

    print(mean_values)
    x_values = [value for _, value in loss_list]
    print(x_values)
    # Plotting
    plt.figure(figsize=(10, 6))

    plt.plot(
        x_values,
        10 * np.log10(mean_values),
        marker="o",
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
    plt.title("Learned stepsize quantization fibbinary", fontsize=14)
    plt.grid(True)
    plt.xlim(2, 11)
    plt.xticks(np.arange(2, 12, 1))
    plt.savefig("loss_lsq_fibbinary_total.png")
