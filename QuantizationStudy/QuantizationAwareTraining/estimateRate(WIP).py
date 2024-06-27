import torch
from torch.utils.data import DataLoader

from pathlib import Path
import warnings
import time

import argparse
import numpy as np

from InformerModel.model import InformerStack
from dataset import SeqData, LoadBatch
from config import get_config
from metrics import NMSELoss, NMSELossSplit

from torch.utils.tensorboard import SummaryWriter

from qtorch.optim import OptimLP
from qtorch.quant import fibonacci_quantize_partial

import numpy as np

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# def get_rate(H, sigma2):
#     rate = np.log2(np.linalg.det(np.eye(2) + 1 / sigma2 * H.T.conj().dot(H)))
#     return rate


# def get_zf_rate(H_hat, H_true, SNR):
#     print(H_hat.shape)
#     print(H_true.shape)
#     D_np = get_zf_precoder(H_hat)  # Calculate the ZF precoder shape = [256,2,4]
#     print(D_np.shape)  # [256, 4, 2]
#     HF = np.matmul(H_true, D_np)
#     print(HF.shape)
#     rate = SNR_rate(HF, SNR)
#     return rate


# def get_zf_precoder(H_hat):
#     D_np = np.linalg.pinv(H_hat)  # shape = [64, 2, 4]
#     print(D_np.shape)
#     D_np = D_np / np.linalg.norm(D_np, axis=(2), keepdims=True)
#     return D_np


# def SNR_rate(H, SNR):
#     H_conj = np.transpose(H.resolve_conj().numpy(), (0, 2, 1))
#     eye = np.eye(2)
#     print(eye.shape)

#     # Ensure matrix multiplication results in a 2x2 matrix
#     H_conj_H = np.matmul(H_conj, H)
#     print(H_conj_H.shape)
#     print(H.shape)
#     print(H_conj.shape)
#     # print((eye + (10 ** (SNR / 10))) * H_conj_H)
#     print(np.linalg.det((eye + (10 ** (SNR / 10))) * H_conj_H))

#     # Calculate the rate using the mean of the determinant of the 2x2 matrices
#     rate = np.mean(np.log2(np.linalg.det(eye + (10 ** (SNR / 10)) * H_conj_H)))
#     return rate


# def SINR_rate(HF, SNR):
#     HF = torch.pow(torch.abs(torch.from_numpy(HF).to(device)), 2)
#     HF_diag = HF * torch.eye(2, device=device)
#     rate = torch.mean(
#         torch.sum(
#             torch.log2(
#                 1
#                 + torch.sum(HF_diag, 2)
#                 / (torch.abs(torch.sum(HF - HF_diag, 2)) + 1 / (10 ** (SNR / 10)))
#             ),
#             1,
#         )
#     )
#     return rate


def get_rate(H, sigma2):
    rate = np.log2(np.linalg.det(np.eye(2) + 1 / sigma2 * H.T.conj().dot(H)))
    return rate


def get_zf_rate(H_hat, H_true, SNR):
    D_np = get_zf_precoder(H_hat)
    HF = np.matmul(D_np, H_true)
    rate = SNR_rate(HF, SNR)
    return rate


def get_zf_precoder(H_hat):
    D_np = np.linalg.pinv(H_hat)  # shape = [64, 2, 4]
    D_np = D_np / np.linalg.norm(D_np, axis=(2), keepdims=True)
    return D_np


def SNR_rate(H, SNR):
    rate = np.mean(
        np.log2(
            np.linalg.det(
                np.eye(2)
                + (10 ** (SNR / 10)) * np.matmul(H.conj().transpose(0, 2, 1), H)
            )
        )
    )
    return rate


def SINR_rate(HF, SNR):
    HF = torch.pow(torch.abs(torch.from_numpy(HF).cuda()), 2)
    HF_diag = HF * torch.eye(2).cuda()
    rate = torch.mean(
        torch.sum(
            torch.log2(
                1
                + torch.sum(HF_diag, 2)
                / (torch.abs(torch.sum(HF - HF_diag, 2)) + 1 / (10 ** (SNR / 10)))
            ),
            1,
        )
    )
    return rate


def real2complex(data):
    B, P, N = data.shape
    data2 = data.reshape([B, P, N // 2, 2])
    data2 = data2[:, :, :, 0] + 1j * data2[:, :, :, 1]
    return data2


def get_dataset(config):
    trainDatasetName = f'../GenerateDatasets/Datasets/{config["dataset_name"]}.pickle'
    evaluateDatasetName = (
        f'../GenerateDatasets/Datasets/{config["dataset_name"]}__validate.pickle'
    )

    trainData = SeqData(
        trainDatasetName, config["seq_len"], config["pred_len"], config["SNR"]
    )
    trainLoader = DataLoader(
        dataset=trainData,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    evaluateData = SeqData(
        evaluateDatasetName, config["seq_len"], config["pred_len"], config["SNR"]
    )
    evaluaterLoader = DataLoader(
        dataset=evaluateData,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    return trainLoader, evaluaterLoader


def get_model(config):

    model = InformerStack(
        config["enc_in"],
        config["dec_in"],
        config["c_out"],
        config["seq_len"],
        config["label_len"],
        config["pred_len"],
        config["factor"],
        config["d_model"],
        config["n_heads"],
        config["e_layers"],
        config["d_layers"],
        config["d_ff"],
        config["dropout"],
        config["attn"],
        config["embed"],
        config["activation"],
        config["output_attention"],
        config["distil"],
        device,
    )
    model = model.cuda() if torch.cuda.is_available() else model
    return model


def run_validation(model, val_dataloader, device):
    # Set the model to evaluation mode
    model.eval()
    # Create a metric to store the loss
    rate = 0
    loss_fn = NMSELossSplit()
    loss = torch.zeros(5).to(device)

    # Iterate over the validation dataloader
    for batch in val_dataloader:
        H, H_noise, H_seq, H_pred = batch
        data, label = LoadBatch(H_seq), LoadBatch(H_pred)

        label = label.to(device)
        encoder_input = data.to(device)
        decoder_input = torch.zeros_like(encoder_input[:, -config["pred_len"] :, :]).to(
            device
        )
        decoder_input = torch.cat(
            [
                encoder_input[
                    :, config["seq_len"] - config["label_len"] : config["seq_len"], :
                ],
                decoder_input,
            ],
            dim=1,
        )
        with torch.no_grad():
            output, attn = model(
                encoder_input,
                range(config["seq_len"]),
                decoder_input,
                range(config["pred_len"] + config["label_len"]),
            )
        loss += loss_fn(output, label)
        # H = LoadBatch(H)
        # H_noise = LoadBatch(H_noise)
        output = real2complex(np.array(output))
        output = torch.tensor(
            output.reshape(config["batch_size"], config["pred_len"], 2, 4)
        )
        H_hat = output[:, -1, :, :].transpose(1, 2).numpy()
        H_true = H_pred[:, -1, :, :].transpose(1, 2).numpy()

        rate += get_zf_rate(H_hat, H_true, config["SNR"])

    return rate / len(val_dataloader), loss / len(val_dataloader)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()

    # Define the device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(
            f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB"
        )
    elif device == "mps":
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print(
            "      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc"
        )
        print(
            "      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu"
        )
    device = torch.device(device)

    train_dataloader, val_dataloader = get_dataset(config)

    model = get_model(config)

    for num_bits in range(2, 9):

        config["model_filename"] = f"weights/wl{num_bits}_fl{num_bits-1}_epoch_99.pt"

        print(f"Preloading model {config['model_filename']}")
        if torch.cuda.is_available():
            state = torch.load(config["model_filename"], map_location=device)
        else:
            state = torch.load(config["model_filename"], map_location="cpu")
        model.load_state_dict(state["model_state_dict"])

        rate, loss = run_validation(model, val_dataloader, device)

        print(f"Bitwidth: {num_bits}    Rate: {rate}    loss: {loss}")
