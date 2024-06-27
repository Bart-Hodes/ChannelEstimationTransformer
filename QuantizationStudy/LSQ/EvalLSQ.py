import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from pathlib import Path
import warnings
import time

import argparse
import numpy as np
from config import get_config

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.InformerLSQ.model import InformerStack
from models.InformerLSQ.LSQ import LinearLSQ, Conv1dLSQ
from Utils.dataset import SeqData, LoadBatch
from Utils.metrics import NMSELoss, NMSELossSplit

from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with custom config")
    parser.add_argument("--d_model", type=int, help="Dimension of model", default=None)
    parser.add_argument(
        "--d_ff", type=int, help="Dimension of feed forward layer", default=None
    )
    parser.add_argument("--seq_len", type=int, help="Sequence length", default=None)
    parser.add_argument("--label_len", type=int, help="Label length", default=None)
    parser.add_argument("--attn", type=str, help="attn", default=None)
    parser.add_argument("--distil", type=bool, help="distil", default=None)
    parser.add_argument("--SNR", type=int, help="SNR", default=None)
    return parser.parse_args()


def get_dataset(config):
    trainDatasetName = f'../GenerateDatasets/Datasets/{config["dataset_name"]}.pickle'
    evaluateDatasetName = (
        f'../GenerateDatasets/Datasets/{config["dataset_name"]}__validate.pickle'
    )

    trainData = SeqData(
        trainDatasetName, config["seq_len"], config["pred_len"], SNR=config["SNR"]
    )
    trainLoader = DataLoader(
        dataset=trainData,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    evaluateData = SeqData(
        evaluateDatasetName, config["seq_len"], config["pred_len"], SNR=config["SNR"]
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
        config["device"],
        config["num_bits"],
    )
    model = model.cuda() if torch.cuda.is_available() else model
    print("Getting model")
    # Iterate over all modules in the model
    if config["num_bits"] is not None:
        print(f"Quantizing model to {config['num_bits']} bits")
        for name, module in model.named_modules():
            if isinstance(module, LinearLSQ):
                module.quantize = True
                module.nbits = int(config["num_bits"])
                module.reset_parameters()
            if isinstance(module, Conv1dLSQ):
                module.quantize = True
                module.nbits = int(config["num_bits"])
                module.reset_parameters()
    return model


def run_validation(model, val_dataloader, device):
    # Set the model to evaluation mode
    model.eval()
    # Create a metric to store the loss
    loss_fn = NMSELossSplit()
    loss = torch.zeros(5).to(device)

    # Iterate over the validation dataloader
    for batch_idx, batch in enumerate(val_dataloader):
        # for batch in val_dataloader:
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

    val_loss = loss / len(val_dataloader)
    return val_loss


import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()

    print(len(sys.argv))

    # Add network settings to the config
    config.update(
        {
            "device": (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            ),
        }
    )

    # Remove unwanted characters and replace them with underscores
    e_layers_str = "_".join(map(str, config["e_layers"]))
    # config['attn'] = config['attn'].replace(" ", "_")
    # config['embed'] = config['embed'].replace(" ", "_")
    # config['activation'] = config['activation'].replace(" ", "_")

    model_name = f"{config['enc_in']}_{config['dec_in']}_{config['c_out']}_{config['seq_len']}_{config['label_len']}_{config['pred_len']}_{config['factor']}_{config['d_model']}_{config['n_heads']}_{e_layers_str}_{config['d_layers']}_{config['d_ff']}_{config['dropout']}_{config['attn']}_{config['embed']}_{config['activation']}"
    print("Model_name: ", model_name)
    print("experiment_name: ", config["experiment_name"])

    trainLoader, evaluateLoader = get_dataset(config)

    loss_list = []
    for nbits in [2, 3, 4, 5, 6, 7, 8]:
        config["num_bits"] = nbits
        print(f"Training with {nbits} bits")

        model = get_model(config)
        model_filename = f"weights/nbits{config['num_bits']}_epoch_199.pt"

        if torch.cuda.is_available():
            state = torch.load(model_filename, map_location="cuda:0")
        else:
            state = torch.load(model_filename, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])

        loss = run_validation(model, evaluateLoader, config["device"])
        print(loss)
        data_np = [[tensor.cpu().numpy()] for tensor in loss]

        loss_list.append([data_np, nbits])

    # Save the loss list to a file
    import pickle

    # print("Loss list: ", loss_list)

    # # Save the loss list to a file using pickle
    # with open(f"loss_list.pickle", "wb") as f:
    #     pickle.dump(loss_list, f)
