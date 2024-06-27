import matplotlib.pyplot as plt
import scienceplots

# Use the science plots style
plt.style.use(["science", "no-latex"])


def generate_fibbinary_numbers(max_val):
    """Generate all fibbinary numbers up to max_val."""
    fibbinary_numbers = []
    for i in range(max_val + 1):
        bin_rep = bin(i)[2:]  # Get binary representation of i
        if "11" not in bin_rep:
            fibbinary_numbers.append(i)
    return fibbinary_numbers


def quantize_to_fibbinary(n, fibbinary_numbers):
    """Quantize n to the nearest fibbinary number in the list."""
    closest = min(fibbinary_numbers, key=lambda x: abs(x - n))
    return closest


# Define the range for the x-axis (both positive and negative)
max_val = 64
x_values_positive = list(range(max_val + 1))
x_values_negative = list(range(-max_val, 0))

# Generate fibbinary numbers up to max_val
fibbinary_numbers = generate_fibbinary_numbers(max_val)

# Quantize each x value to the nearest fibbinary number
y_values_positive = [
    quantize_to_fibbinary(x, fibbinary_numbers) for x in x_values_positive
]
y_values_negative = [
    -quantize_to_fibbinary(-x, fibbinary_numbers) for x in x_values_negative
]

# Combine positive and negative values for plotting
x_values = x_values_negative + x_values_positive
y_values = y_values_negative + y_values_positive

# Create the plot
plt.figure(figsize=(6.4, 4.8))
plt.step(x_values, y_values, where="mid", color="b", linewidth=2)
plt.title("Fibbinary Quantization")
plt.xlabel("Original Number")
plt.ylabel("Quantized Fibbinary Number")
plt.grid(True)
plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)
plt.xlim(-64, 64)
plt.ylim(-64, 64)


plt.savefig("FibbinaryMapping.png", dpi=300)
