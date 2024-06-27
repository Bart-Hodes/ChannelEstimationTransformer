import pickle
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["ieee", "no-latex"])

speedup = 3


if __name__ == "__main__":
    with open("../runtime2.pickle", "rb") as f:
        data = pickle.load(f)

    print(data)

    plt.figure(figsize=(6.4, 4.8))
    #
    colors = [
        "#0C5DA5",
        "#00B945",
        "#FF9500",
        "#FF2C00",
        "#845B97",
        "#474747",
        "#9e9e9e",
    ]

    for value in data:
        print(value)
        print(value[0][0])
        color_idx = colors[value[0][1]]
        plt.scatter(
            value[0][0],
            value[1][0],
            marker="o",
            color=color_idx,
        )

        plt.annotate(
            f"{value[0][0]},{value[0][1]}",
            (value[0][0], value[1][0]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
        )

    for srs in range(1, 6):
        print(srs * 0.625 * speedup)
        plt.plot(
            [0, 8],
            [srs * 0.625 * speedup, srs * 0.625 * speedup],
            label=f"Number of encoders: {srs}",
            color=colors[srs - 1],
            linestyle="--",
        )

    plt.legend()
    plt.xlabel("Number of encoders")
    plt.ylabel("Inference Time (ms)")
    plt.title("Influence of Hyperparameters on Runtime")
    plt.xlim([0.5, 5.5])
    plt.grid(True)
    plt.xticks(range(6))  # Assuming there are 5 layers, adjust if needed
    plt.savefig("EncoderDecoderInferTime.png", dpi=300)
    plt.show()
