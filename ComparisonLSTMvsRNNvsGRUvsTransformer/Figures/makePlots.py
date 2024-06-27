import pickle
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "no-latex"])

if __name__ == "__main__":
    with open("../loss_GRU.pkl", "rb") as f:
        loss_GRU = pickle.load(f)
    with open("../loss_LSTM.pkl", "rb") as f:
        loss_LSTM = pickle.load(f)
    with open("../loss_RNN.pkl", "rb") as f:
        loss_RNN = pickle.load(f)
    with open("../loss_Transformer.pkl", "rb") as f:
        loss_Transformer = pickle.load(f)
    with open("../loss_Informer.pkl", "rb") as f:
        loss_Informer = pickle.load(f)

    print("Loss GRU:", loss_GRU)
    print("Loss LSTM:", loss_LSTM)
    print("Loss RNN:", loss_RNN)
    print("Loss Transformer:", loss_Transformer)
    print("Loss Informer:", loss_Informer)

    plt.figure(figsize=(6.4, 4.8))
    for loss in [loss_GRU, loss_LSTM, loss_RNN, loss_Transformer, loss_Informer]:
        y, x = zip(*loss)
        y = np.array(y)
        print(y[:, -1])
        plt.plot(x, 10 * np.log10(y[:, -1]), marker="o")

    plt.legend(["GRU", "LSTM", "RNN", "Transformer", "Informer"])
    plt.xlabel("Signal to Noise Ratio (dB)")
    plt.title(
        "Loss Comparison of different ANN's for prediction length 5 at different SNR"
    )
    plt.ylabel("NMSE Loss (dB)")
    plt.grid(True)
    plt.xticks([12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    plt.xlim([12, 21])
    plt.savefig("loss_comparison.png", dpi=300)

    ####################################################################################################################################

    plt.figure(figsize=(6.4, 4.8))
    for loss in [loss_GRU, loss_LSTM, loss_RNN, loss_Transformer, loss_Informer]:
        y, x = zip(*loss)
        plt.plot([1, 2, 3, 4, 5], 10 * np.log10(y[:][-1]), marker="o")

    plt.legend(["GRU", "LSTM", "RNN", "Transformer", "Informer"])
    plt.xlabel("SRS (0.625 ms)")
    plt.title(f"Loss Comparison of different neural networks at SNR {x[-1]} dB")
    plt.ylabel("NMSE Loss (dB)")
    plt.xticks([1, 2, 3, 4, 5])
    plt.xlim([1, 5])
    plt.grid(True)
    plt.savefig("loss_pred_len.png", dpi=300)
