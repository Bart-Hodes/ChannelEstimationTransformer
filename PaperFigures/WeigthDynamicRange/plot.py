import json
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

# Use the science plots style
plt.style.use(["science", "no-latex"])

# Load weights from the first JSON file
with open(
    "weight_export/decoder.layers.2.self_attention.value_projection.weight.json", "r"
) as file1:
    weights1 = json.load(file1)

# Load weights from the second JSON file
with open("weight_export/decoder.layers.0.conv2.weight.json", "r") as file2:
    weights2 = json.load(file2)

# 2D Bar Plot
plt.figure(figsize=(6.4, 4.8))  # Adjusted figure size for side by side plots

# Plot the first set of weights
plt.subplot(2, 1, 1)
xs1 = np.arange(len(weights1))
ys1 = [weight[0] for weight in weights1]
plt.bar(xs1, ys1, width=0.4)
plt.xlabel("Token")
plt.ylabel("Absolute Weight Value")
plt.legend()
plt.ylim(-0.25, 0.25)
plt.xlim(0, 128)
plt.title("Decoder 2 - self attention - value projection - weights")


# print([f"index {i}: {x}" for x, i in enumerate(weights2)])

# Plot the second set of weights
plt.subplot(2, 1, 2)
xs2 = np.arange(len(weights2[1]))
ys2 = [weight[0] for weight in weights2[117]]
# print(ys2)
plt.bar(xs2, ys2, width=0.4)
plt.xlabel("Token")
plt.ylabel("Absolute Weight Value")
plt.legend()
plt.ylim(-2.5, 2.5)
plt.xlim(0, 64)
plt.title("Decoder 0 - Conv2 - weights")

plt.tight_layout()  # Adjust subplot parameters to give specified padding
plt.savefig("Weights2d_side_by_side.png", dpi=300)
plt.show()
