import pickle
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["ieee", "no-latex"])


def normalize_runtimes(data):
    normalized_data = {}
    for entry in data:
        key = "_".join(entry[0].split("_")[:-1])
        if key not in normalized_data:
            normalized_data[key] = {"values": [], "std": []}
        normalized_data[key]["values"].append(entry[1][0])
        normalized_data[key]["std"].append(entry[1][1])

    normalized_results = {}
    for key, values in normalized_data.items():
        max_runtime = max(values["values"])
        max_runtime = 1
        normalized_values = [value / max_runtime for value in values["values"]]
        normalized_std = [std / max_runtime for std in values["std"]]
        normalized_results[key] = {"values": normalized_values, "std": normalized_std}

    return normalized_results


if __name__ == "__main__":
    with open("../runtime2.pickle", "rb") as f:
        data = pickle.load(f)

    normalized_data = normalize_runtimes(data)
    print(normalized_data)

    num_plots = len(normalized_data)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate number of rows needed

    color = [
        "#0C5DA5",
        "#00B945",
        "#FF9500",
        "#FF2C00",
        "#845B97",
        "#474747",
        "#9e9e9e",
        "#BBBBBB",
    ]

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6.4, 6.4))
    fig.suptitle("Influence of Hyperparameters on Runtime (2080TI)", fontsize=16)
    fig.supylabel("Inference Time (ms)", fontsize=14)
    # Marker styles for different categories
    axes = axes.flatten()  # Flatten in case of single row or column

    for idx, (key, values) in enumerate(normalized_data.items()):
        x_values = range(len(values["values"]))
        ax = axes[idx]

        print(key, values)

        if key == "e_layers":
            x_values = [1, 2, 3, 4, 5]
            key = "Number of encoders layers"
            ax.set_xlim(0, 6)
            ax.set_ylim(3, 9)
        if key == "d_layers":
            x_values = [1, 2, 3, 4, 5]
            key = "Number of decoder layers"
            ax.set_xlim(0, 6)
            ax.set_ylim(4, 10)
        if key == "n_heads":
            x_values = [1, 2, 3, 4, 5]
            key = "Number of attention heads"
            ax.set_xlim(0, 6)
            ax.set_ylim(6, 12)
        if key == "d_ff":
            x_values = [64, 128, 256, 512, 1024]
            key = "Feedforward dimension"
            ax.set_xlim(0, 1200)
            ax.set_ylim(6, 12)
        if key == "d_model":
            x_values = [64, 128, 256, 512, 1024]
            key = "Model dimension"
            ax.set_xlim(0, 1200)
            ax.set_ylim(6, 12)
        if key == "seq_len":
            x_values = [12, 24, 48, 60, 72]
            key = "Sequence length"
            ax.set_xlim(0, 80)
            ax.set_ylim(6, 12)
        if key == "pred_len":
            x_values = [1, 3, 5, 7, 9]
            ax.set_xlim(0, 10)
            ax.set_ylim(6, 12)
            key = "Prediction length"
        if key == "label_len":
            x_values = [5, 10, 15, 20, 25]
            ax.set_xlim(0, 30)
            ax.set_ylim(6, 12)
            key = "Label length"

        ax.errorbar(
            x_values,
            values["values"],
            yerr=values["std"],
            label=key,
            fmt="o",
            color=color[idx],
            capsize=5,
            alpha=0.8,
        )
        # ax.set_title(key)
        ax.set_xlabel(key)
        # ax.set_ylabel("Normalized Inference Time")
        ax.grid(True)

    # Remove any empty subplots
    for j in range(idx + 1, num_rows * num_cols):
        fig.delaxes(axes[j])

    fig.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to fit the suptitle
    plt.savefig("normalized_error_bar_plot.png", dpi=300)
    plt.show()
