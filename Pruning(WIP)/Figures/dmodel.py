from parse import parse_log_file
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    log_file = "dmodel.txt"
    results = parse_log_file(log_file)
    print(results)
    # Plotting
    plt.figure(figsize=(10, 6))
    for model, loss_pred_len_list in results.items():
        plt.plot(10 * np.log(loss_pred_len_list), label=model)

    plt.xlabel("Epoch")
    plt.ylabel("Loss NMSE")
    plt.title("Loss vs dmodel SNR:40")
    plt.legend(["dmodel:64", "dmodel:96", "dmodel:128"])
    plt.grid(True)
    plt.savefig("loss_VS_dmodel.png")
