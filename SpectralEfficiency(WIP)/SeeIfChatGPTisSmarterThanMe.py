import numpy as np
import matplotlib.pyplot as plt


# Function to calculate the spectral efficiency
def spectral_efficiency(H, SNR_dB):
    SNR = 10 ** (SNR_dB / 10)
    I = np.eye(H.shape[1])
    rate = np.mean(
        np.log2(np.linalg.det(I + SNR * np.matmul(H.conj().transpose(0, 2, 1), H)))
    )
    return rate


# Function to calculate the NMSE
def nmse(H, H_pred):
    error = np.linalg.norm(H - H_pred, axis=(1, 2)) ** 2
    norm = np.linalg.norm(H, axis=(1, 2)) ** 2
    return np.mean(error / norm)


# Generating a random channel matrix H
np.random.seed(0)
H = np.random.randn(100, 2, 2) + 1j * np.random.randn(100, 2, 2)

# Define different NMSE levels
nmse_levels = np.linspace(0.01, 1, 100)

# SNR value for the plot
SNR_dB = 21  # Fixed SNR in dB

nmse_values = []
spectral_eff_values = []

for nmse_target in nmse_levels:
    # Generate a predicted H matrix with the desired NMSE
    noise_std_dev = np.sqrt(
        nmse_target * np.linalg.norm(H, axis=(1, 2)) ** 2 / (H.shape[1] * H.shape[2])
    )
    noise = noise_std_dev[:, np.newaxis, np.newaxis] * (
        np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape)
    )
    H_pred = H

    current_nmse = nmse(H, H_pred)
    print(current_nmse)
    nmse_values.append(current_nmse)
    spectral_eff_values.append(spectral_efficiency(H_pred, SNR_dB))

# Plotting NMSE vs Spectral Efficiency
plt.figure(figsize=(10, 6))
plt.plot(nmse_values, spectral_eff_values, marker="o", label=f"SNR = {SNR_dB} dB")
plt.xlabel("NMSE")
plt.ylabel("Spectral Efficiency (bps/Hz)")
plt.title("NMSE vs Spectral Efficiency")
plt.legend()
plt.grid(True)
plt.savefig("NMSE_vs_Spectral_Efficiency.png")
