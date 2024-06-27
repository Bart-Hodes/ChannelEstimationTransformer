import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("partial_proximal_dumbway.csv")
df_non_uniform = pd.read_csv("partial_proximal.csv")

previous_step = -1
previous_time = 0
for index, row in df.iterrows():
    if row["Step"] < previous_step:
        multiplier = (previous_step - row["Step"]) // 10 + 1
        df.loc[index:, "Step"] += 10 * multiplier
    previous_step = row["Step"]
    print(row["Wall time"] - previous_time)
    print(row)
    previous_time = row["Wall time"]

previous_step = -1
for index, row in df_non_uniform.iterrows():
    if row["Step"] < previous_step:
        multiplier = (previous_step - row["Step"]) // 10 + 1
        df_non_uniform.loc[index:, "Step"] += 10 * multiplier
    previous_step = row["Step"]
print(df)

# Interpolate to ensure steps from 0 to 179
df = df.set_index("Step").reindex(range(179)).interpolate(method="linear").reset_index()
df_non_uniform = (
    df_non_uniform.set_index("Step")
    .reindex(range(179))
    .interpolate(method="linear")
    .reset_index()
)

# Plot the data
plt.plot(
    df["Step"],
    df["Value"],
    linestyle="-",
    label="Uniform",
)

plt.plot(
    df_non_uniform["Step"],
    df_non_uniform["Value"],
    linestyle="-",
    label="Non-Uniform",
)

plt.savefig("StepsizeStratagiesDebug.png", dpi=300)

df.to_csv("partial_proximal_dumbway_fixed.csv", index=False)
df_non_uniform.to_csv("partial_proximal_fixed.csv", index=False)
