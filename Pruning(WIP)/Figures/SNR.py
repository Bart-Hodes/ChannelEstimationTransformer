from parse import parse_log_file
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    log_file = "SNR.txt"
    results = parse_log_file(log_file)
    print(results.items())
    # Plotting
    plt.figure(figsize=(10, 6))
    for model, loss_pred_len_list in results.items():
        print(loss_pred_len_list)
        plt.plot(10 * np.log(loss_pred_len_list), label=model)

    plt.xlabel("Epoch")
    plt.ylabel("Loss NMSE")
    plt.title("Loss vs SNR SNR:40")
    plt.legend(["SNR:12", "SNR:14", "SNR:16", "SNR:18", "SNR:20"])
    plt.grid(True)
    plt.savefig("loss_VS_SNR.png")
