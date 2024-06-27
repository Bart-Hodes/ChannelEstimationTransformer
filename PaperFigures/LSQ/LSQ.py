import matplotlib.pyplot as plt
import numpy as np
import pickle
import scienceplots


if __name__ == "__main__":
    # Load the loss_list from the pickle file
    with open("loss_list.pickle", "rb") as f:
        loss_list = pickle.load(f)

    # split over prediction length
    mean_values = [10 * np.log10(np.array(arr).flatten()[4]) for arr, _ in loss_list]
    x_values = [value for _, value in loss_list]
    # Plotting
    plt.style.use(["science", "no-latex"])
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(
        x_values,
        mean_values,
        marker="o",
        label="Standard",
    )

    with open("loss_list_fibbinary.pickle", "rb") as f:
        loss_list = pickle.load(f)

    # split over prediction length
    mean_values = [10 * np.log10(np.array(arr).flatten()[4]) for arr, _ in loss_list]
    print(mean_values)

    x_values = [value for _, value in loss_list]
    # Plotting

    plt.plot(
        x_values,
        mean_values,
        marker="o",
        label="Fibbinary",
    )

    plt.axhline(
        y=10 * np.log10(0.3466),
        color="black",
        linestyle="--",
        label=f"FP32 Accuracy",
    )

    plt.xlabel("Number of bits", fontsize=14)
    plt.ylabel("Loss NMSE dB", fontsize=14)
    plt.legend()
    plt.title("Learned stepsize quantization", fontsize=14)
    plt.grid(True)
    plt.xlim(2, 8)
    # plt.ylim(-15, -8)
    plt.savefig("loss_lsq_total.png")
