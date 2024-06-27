from parse import parse_log_file
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    log_file = "distil.txt"
    results = parse_log_file(log_file)
    print(results)
    # Plotting
    plt.figure(figsize=(10, 6))
    for model, loss_pred_len_list in results.items():
        plt.plot(10 * np.log(loss_pred_len_list), label=model)

    plt.xlabel("Epoch")
    plt.ylabel("Loss NMSE")
    plt.title("Loss vs distill SNR:40")
    plt.legend(["distill:True", "distill:False"])
    plt.grid(True)
    plt.savefig("loss_VS_distill.png")