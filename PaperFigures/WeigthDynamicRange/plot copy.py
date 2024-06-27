import torch
import json
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# Load weights
with open("encoder.encoders.0.attn_layers.0.conv2.weight.json", "r") as file:
    weights = json.load(file)

    # print(weights)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for i in range(0, len(weights)):
        xs = np.arange(len(weights[i]))
        ys = [weight[0] for weight in weights[i]]
        # print(xs)
        # print(ys)
        ax.bar(xs, ys, zs=i, zdir="y")  # Convert weights[i] to numpy array

    plt.savefig("Weights.png", dpi=300)
