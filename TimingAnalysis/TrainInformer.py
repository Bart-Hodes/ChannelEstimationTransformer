import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from pathlib import Path
import warnings
import time

import argparse
import numpy as np

from Informer.model import InformerStack
from dataset import SeqData, LoadBatch
from config import get_config
from metrics import NMSELoss, NMSELossSplit

from torch.utils.tensorboard import SummaryWriter

import datetime
import pickle

current_datetime = datetime.datetime.now()
print("Current Date and Time:", current_datetime)


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_dataset(config):
    trainDatasetName = (
        f'../GenerateDatasets/Datasets/backup/{config["dataset_name"]}.pickle'
    )
    evaluateDatasetName = (
        f'../GenerateDatasets/Datasets/backup/{config["dataset_name"]}__validate.pickle'
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
    loss_fn = NMSELossSplit()
    loss = torch.zeros(5).to(device)
    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )

    repetitions = 1000
    timings = np.zeros((repetitions, 1))

    # Iterate over the validation dataloader
    for batch_idx, batch in enumerate(val_dataloader):
        if batch_idx >= repetitions + 20:
            break

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
            starter.record()
            output, attn = model(
                encoder_input,
                range(config["seq_len"]),
                decoder_input,
                range(config["pred_len"] + config["label_len"]),
            )
            ender.record()

            if batch_idx > 20:
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[batch_idx - 20] = curr_time
                # print("Batch: ", batch_idx, "Time: ", curr_time, "ms", flush=True)

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)

    print(f"Mean time: {mean_syn} ms", flush=True)
    print(f"Std time: {std_syn} ms", flush=True)

    loss = (mean_syn, std_syn)

    return loss


def train_model(config):
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

    # Make sure the weights folder exists
    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader = get_dataset(config)

    model = get_model(config)

    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config["num_epochs"], eta_min=0, last_epoch=-1
    )

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0

    # Run validation at the end of every epoch
    val_loss = run_validation(model, val_dataloader, device)

    return val_loss, model


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()

    model_name = f"Informer_ei_{config['enc_in']}_di_{config['dec_in']}_co_{config['c_out']}_sl_{config['seq_len']}_ll_{config['label_len']}_pl_{config['pred_len']}_f_{config['factor']}_dm_{config['d_model']}_nh_{config['n_heads']}_el_{config['e_layers']}_dl_{config['d_layers']}_df_{config['d_ff']}_do_{config['dropout']}_at_{config['attn']}_em_{config['embed']}_ac_{config['activation']}"
    print("Model_name: ", model_name)

    config["model_folder"] = config["model_folder"] + "/Informer"
    config["experiment_name"] = (
        "/".join(config["experiment_name"].split("/", 1)) + "/Informer" + model_name
    )
    config["model_basename"] = config["model_basename"] + model_name
    runtime = []

    # for e_layers in [1, 2, 3, 4, 5]:
    #     for d_layers in [1, 2, 3, 4, 5]:
    #         config["e_layers"] = [e_layers]
    #         config["d_layers"] = d_layers

    #         val_loss, model = train_model(config)
    #         runtime.append([[e_layers, d_layers], val_loss])

    for e_layers in [1, 2, 3, 4, 5]:
        config["e_layers"] = [e_layers]
        val_loss, model = train_model(config)
        runtime.append([f"e_layers_{e_layers}", val_loss])

    for d_layers in [1, 2, 3, 4, 5]:
        config["d_layers"] = d_layers
        val_loss, model = train_model(config)
        runtime.append([f"d_layers_{d_layers}", val_loss])

    for n_heads in [1, 2, 3, 4, 5]:
        config["n_heads"] = n_heads
        val_loss, model = train_model(config)
        runtime.append([f"n_heads_{n_heads}", val_loss])

    for d_ff in [64, 128, 256, 512, 1024]:
        config["d_ff"] = d_ff
        val_loss, model = train_model(config)
        runtime.append([f"d_ff_{d_ff}", val_loss])

    for d_model in [64, 128, 256, 512, 1024]:
        config["d_model"] = d_model
        val_loss, model = train_model(config)
        runtime.append([f"d_model_{d_model}", val_loss])

    for seq_len in [12, 24, 48, 60, 72]:
        config["seq_len"] = seq_len
        val_loss, model = train_model(config)
        runtime.append([f"seq_len_{seq_len}", val_loss])

    for pred_len in [1, 3, 5, 7, 9]:
        config["pred_len"] = pred_len
        val_loss, model = train_model(config)
        runtime.append([f"pred_len_{pred_len}", val_loss])

    for label_len in [5, 10, 15, 20, 25]:
        config["label_len"] = label_len
        val_loss, model = train_model(config)
        runtime.append([f"label_len_{label_len}", val_loss])

print(runtime)

# Save runtime as pickle object
with open("runtime2.pickle", "wb") as f:
    pickle.dump(runtime, f)
