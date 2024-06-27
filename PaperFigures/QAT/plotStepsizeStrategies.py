import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
import numpy as np

# Use the science plots style
plt.style.use(["science", "no-latex"])
plt.figure(figsize=(6.4, 4.8))

# Read the CSV file into DataFrames
df_non_uniform = pd.read_csv("Data/partial_proximal_fixed.csv")
df_uniform = pd.read_csv("Data/partial_proximal_dumbway_fixed.csv")

# Convert the 'Value' column to dB
df_non_uniform["Value_dB"] = 10 * np.log10(df_non_uniform["Value"])
df_uniform["Value_dB"] = 10 * np.log10(df_uniform["Value"])

# Plot the data
plt.plot(
    df_non_uniform["Step"],
    df_non_uniform["Value_dB"],
    linestyle="-",
    label="Non-uniform",
)

plt.plot(
    df_uniform["Step"],
    df_uniform["Value_dB"],
    linestyle="-",
    label="Uniform",
)

# Customize the plot
plt.title("Incremental QAT nearest-round with nearest weights first")
plt.xlabel("Epoch")
plt.ylabel("NMSE (dB)")
plt.legend()
plt.xlim(0, 180)
plt.grid(True)

# Save the plot as a PNG file
plt.savefig("StepsizeStratagies.png", dpi=300)
plt.show()
