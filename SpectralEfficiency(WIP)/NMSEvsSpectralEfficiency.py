import numpy as np
from dataset import SeqData, LoadBatch
from torch.utils.data import DataLoader
from metrics import NMSELoss, NMSELossSplit


def get_dataset(SNR):
    evaluateDatasetName = (
        f"../GenerateDatasets/Datasets/Seq_Len_100_Beamforming_CDLB__validate.pickle"
    )

    evaluateData = SeqData(evaluateDatasetName, 10, 5, SNR=SNR)
    evaluaterLoader = DataLoader(
        dataset=evaluateData,
        batch_size=8,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    return evaluaterLoader


def get_zf_precoder(H_hat):
    D_np = np.linalg.pinv(H_hat)  # shape = [64, 2, 4]
    D_np = D_np / np.linalg.norm(D_np, axis=(1), keepdims=True)
    return D_np


def spectral_efficiency(H_e, H_a, SNR):
    """
    Calculate the spectral efficiency of a SU-mMIMO system.

    Parameters:
    H_e (numpy.ndarray): Estimated channel matrix of shape (batchsize, slot, Tx, Rx).
    H_a (numpy.ndarray): Actual channel matrix of shape (batchsize, slot, Tx, Rx).
    SNR (float): Signal-to-noise ratio.

    Returns:
    float: Spectral efficiency.
    """
    batchsize, slot, Tx, Rx = H_e.shape
    SE = 0.0

    # Iterate over batchsize and slot
    for b in range(batchsize):
        for s in range(slot):
            H_e_bs = H_e[b, s, :, :]
            H_a_bs = H_a[b, s, :, :]

            D_np = get_zf_precoder(H_e_bs)
            H = np.matmul(D_np, H_a_bs)

            # print(D_np.shape)
            # print(H_a_bs.shape)
            # print(H.shape)

            I = np.eye(Rx)
            product = np.matmul(H, H.conj().T)
            # print(product.shape)
            # print(product)
            matrix = I + (10 ** (SNR / 10)) * product
            SE += np.log2(np.linalg.det(matrix))

    # Normalize by the total number of samples
    SE /= batchsize * slot
    return SE


# Example usage
if __name__ == "__main__":
    batchsize = 10
    slot = 5
    Tx = 4
    Rx = 2
    SNR = 21  # Example SNR value
    loss_func = NMSELoss()

    # Generate random estimated and actual channel matrices for demonstration
    H_e = np.random.randn(batchsize, slot, Tx, Rx) + 1j * np.random.randn(
        batchsize, slot, Tx, Rx
    )
    H_a = np.random.randn(batchsize, slot, Tx, Rx) + 1j * np.random.randn(
        batchsize, slot, Tx, Rx
    )

    for NMSE in [0.1, 0.2, 0.3]:
        dataloader = get_dataset(10)
        for batch_idx, batch in enumerate(dataloader):
            H_a, H_noise, H_seq, H_pred = batch

            SE = spectral_efficiency(H_e.numpy(), H_noise.numpy(), SNR)
            print(f"Spectral Efficiency: {SE}")
            if batch_idx == 1:
                break
