import matplotlib.pyplot as plt
import numpy as np
import pickle
import scienceplots
from matplotlib.font_manager import FontProperties

plt.style.use(["science", "no-latex"])
# Define a monospace font property
monospace_font = FontProperties(family="monospace")

max_bits = 9

if __name__ == "__main__":
    # Load the loss_list from the pickle file
    with open("../loss_list_nearest.pkl", "rb") as f:
        loss_list = pickle.load(f)

    loss_nearest = [10 * np.log10(np.array(arr).flatten()) for arr, _ in loss_list]

    print([np.array(arr).flatten() for arr, _ in loss_list])

    # Load the loss_list from the pickle file
    with open("../loss_list_stochastic.pkl", "rb") as f:
        loss_list = pickle.load(f)

    # split over prediction length
    loss_stochastic = [10 * np.log10(np.array(arr).flatten()) for arr, _ in loss_list]

    x_values = [value for _, value in loss_list]

    print(x_values)
    # Plotting
    plt.figure(figsize=(6.4, 4.8))

    floating_point_accuracies = {1: 10 * np.log10(0.0332), 4: 10 * np.log10(0.1091)}
    floating_point_colors = {1: "red", 4: "blue"}

    for i in [1, 4]:
        plot_values_nearest = [arr[i] for arr in loss_nearest]
        plot_values_stochastic = [arr[i] for arr in loss_stochastic]
        plt.plot(
            x_values[0:max_bits],
            plot_values_nearest[0:max_bits],
            marker="o",
            label=f"Nearest-round,    Pred Len: {i + 1}",
        )
        plt.plot(
            x_values[0:max_bits],
            plot_values_stochastic[0:max_bits],
            marker="o",
            label=f"Stochastic-round, Pred Len: {i + 1}",
        )

        # Add horizontal lines for floating point accuracies
        plt.axhline(
            y=floating_point_accuracies[i],
            color=floating_point_colors[i],
            linestyle="--",
            label=f"FP32 Accuracy,    Pred Len: {i + 1}",
        )

    plt.xlabel("Number of bits")
    plt.ylabel("Loss NMSE dB")
    plt.legend(prop=monospace_font)
    plt.ylim(-15, 5)
    plt.xlim(4, 12)
    plt.title("Post training quantization")
    plt.grid(True)
    plt.savefig("loss_ptq.png", dpi=300)
