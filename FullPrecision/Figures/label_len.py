from parse import parse_log_file
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    log_file = "label_length.txt"
    results = parse_log_file(log_file)
    x = np.arange(1, 6, 1)
    # Plotting
    plt.figure(figsize=(10, 6))
    for model, loss_pred_len_list in results.items():
        plt.plot(10 * np.log(loss_pred_len_list), label=model)

    plt.xlabel("Epoch")
    plt.ylabel("Loss NMSE")
    plt.title("Loss vs label length SNR:40")
    plt.legend(["label_len:5", "label_len:10", "label_len:15", "label_len:20"])
    plt.grid(True)
    plt.savefig("loss_VS_label_len.png")
