import matplotlib.pyplot as plt
import pandas as pd
import scienceplots

# Use the science plots style
plt.style.use(["science", "no-latex"])
plt.figure(figsize=(6.4, 4.8))

# Read the CSV file into a DataFrame
df = pd.read_csv("partial_distant_2.csv")
plt.plot(
    df["Step"], df["Value"], marker="o", markersize=2, linestyle="-", label="Distant"
)

df = pd.read_csv("partial_proximal_2.csv")
plt.plot(
    df["Step"], df["Value"], marker="o", markersize=2, linestyle="-", label="Proximal"
)

df = pd.read_csv("partial_stochastic_2.csv")
plt.plot(
    df["Step"], df["Value"], marker="o", markersize=2, linestyle="-", label="Stochastic"
)
# Plot the data


plt.title("temp")
plt.xlabel("Epoch")
plt.ylabel("NMSE")
plt.grid(True)
plt.savefig("partial_distant_2.png", dpi=300)
