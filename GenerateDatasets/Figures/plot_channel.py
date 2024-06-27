import pickle
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import scipy.io as scio

plt.style.use(["science", "no-latex"])

# Load data
data_dict = scio.loadmat("Temp/RF_Channel.mat")
Channel = np.transpose(data_dict["H_channel"], axes=[0, 1, 2, 4, 3])

i = 0
j = 3
print(Channel.shape)
x = np.linspace(0, 20, 2000)
print(x)
plt.plot(x, Channel[0, 0, :, i, j])

print(np.append(Channel[0, 0, 0::100, i, j], 1))
plt.stem(
    np.append(Channel[0, 0, 0::100, i, j], Channel[0, 0, -1, i, j]), basefmt="black"
)
plt.xlabel("SRS (0.625ms)")
plt.ylabel("Real part of the channel")
plt.grid(True)
plt.xlim([0, 20])
plt.ylim(-1.5, 1.5)

plt.savefig("channel.png", dpi=300)
