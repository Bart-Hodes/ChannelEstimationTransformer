import json
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# Use the science plots style
plt.style.use(["science", "no-latex"])
plt.figure(figsize=(6.4, 4.8))

colorcycle = [
    "#0C5DA5",
    "#00B945",
    "#FF9500",
    "#FF2C00",
    "#845B97",
    "#474747",
    "#9e9e9e",
]

# Load weights
with open("encoder.encoders.2.attn_layers.0.conv2.weight.json", "r") as file:
    weights = json.load(file)

    # print(weights)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for i in range(0, len(weights)):
        xs = np.arange(len(weights[i]))
        ys = [weight[0] for weight in weights[i]]
        if np.max(ys) - np.min(ys) < 0.3:
            colorchoice = colorcycle[0]
        else:
            colorchoice = colorcycle[1]
        ax.bar(
            xs, ys, zs=i, zdir="x", color=colorchoice
        )  # Convert weights[i] to numpy array
    ax.set_xlabel("Channel")
    ax.set_ylabel("Token")
    ax.set_zlabel("Weight")
    plt.savefig("Weights.png", dpi=300)
