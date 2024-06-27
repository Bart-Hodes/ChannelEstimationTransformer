from parse import parse_log_file
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    log_file = "sequence_len.txt"
    results = parse_log_file(log_file)
    print(results)
    # Plotting
    x = np.arange(1, 6, 1)
    print(x)
    plt.figure(figsize=(10, 6))
    for model, loss_pred_len_list in results.items():
        plt.plot(x, 10 * np.log(loss_pred_len_list), label=model)

    plt.xlabel("Epoch")
    plt.ylabel("Loss NMSE")
    plt.title("Loss vs sequence length SNR:40")
    plt.legend(
        ["sequence_len:25", "sequence_len:50", "sequence_len:75", "sequence_len:90"]
    )
    plt.grid(True)
    plt.savefig("loss_VS_sequence_len.png")
