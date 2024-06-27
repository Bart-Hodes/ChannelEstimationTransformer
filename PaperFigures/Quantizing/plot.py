import numpy as np
import matplotlib.pyplot as plt

# Parameters
b = 8  # Number of bits
max_value = 2**b - 1
z = 100  # Example value for Z

# Generate histogram data
np.random.seed(0)  # For reproducibility
data = np.random.normal(loc=(2**b - 1) / 2, scale=(2**b - 1) / 6, size=10000)
data = data[(data >= 0) & (data <= max_value)]  # Clip to range

# Create the histogram
hist, bins = np.histogram(data, bins=50, range=(0, max_value), density=True)

# Plot the histogram
plt.bar(bins[:-1], hist, width=bins[1] - bins[0], color="lightblue", alpha=0.5)

# Set x-axis limits and ticks
plt.xlim(0, max_value)
plt.xticks(np.arange(0, max_value + 1, step=(2**b) / 10))

# Annotate Z
plt.axvline(x=z, color="blue", linestyle="dashed")
plt.text(z, max(hist) / 2, "Z", ha="left", color="blue")

# Add labels for 0 and 2^b - 1
plt.text(0, 0, "0", ha="center", va="top")
plt.text(max_value, 0, f"$2^{b} - 1$", ha="center", va="top")

# Add label S
s = (2**b) / 10
plt.annotate("", xy=(s, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="<->"))
plt.text(s / 2, 0, "S", ha="center", va="top")

# Remove y-axis
plt.yticks([])


# Save the figure as an image (replace 'fixed_vs_float.png' with your desired filename)
plt.savefig("fixed_vs_float.png")
