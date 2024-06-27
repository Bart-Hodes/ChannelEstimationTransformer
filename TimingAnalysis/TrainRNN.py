import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from pathlib import Path
import warnings
import time

import argparse
import numpy as np

from models.model import RNN
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

    model = RNN(config["enc_in"], config["enc_in"], config["hs"], config["hl"])

    model = model.cuda() if torch.cuda.is_available() else model
    return model


def run_validation(model, val_dataloader, device):
    # Set the model to evaluation mode
    model.eval()
    # Create a metric to store the loss
    loss_fn = NMSELossSplit()
    loss = torch.zeros(5).to(device)

    # Iterate over the validation dataloader
    for batch_idx, batch in enumerate(val_dataloader):
        H, H_noise, H_seq, H_pred = batch

        data, label = LoadBatch(H_seq), LoadBatch(H_pred)
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            output = model.test_data(data, config["pred_len"], device)
            loss += loss_fn(output, label)

    plt.figure()

    x = np.arange(20, 25, 1)

    for i in range(4):
        for j in range(2):
            plt.subplot(4, 2, i * 2 + j + 1)
            plt.plot(x, label[0, :, (i + j * 4) * 2].cpu().numpy(), label="label")
            plt.plot(x, output[0, :, (i + j * 4) * 2].cpu().numpy(), label="output")
            plt.plot(H[0, -25:, j, i].cpu().numpy(), label="input")
    plt.savefig("output_eval_RNN.png")
    plt.close()
    return loss / len(val_dataloader)


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

    print("Learning rate: ", config["lr"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config["num_epochs"], eta_min=0, last_epoch=-1
    )

    # optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0

    model_filename = (
        f"{config['model_folder']}/{config['model_basename']}{config['preload']}.pt"
    )

    if config["preload"] is not None:
        print(f"Preloading model {model_filename}")
        if torch.cuda.is_available():
            state = torch.load(model_filename, map_location=device)
        else:
            state = torch.load(model_filename, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = NMSELoss()
    loss_fn_debug = NMSELossSplit()

    for epoch in range(config["num_epochs"]):
        epoch_start_time = time.time()
        torch.cuda.empty_cache()
        model.train()
        total_loss = 0.0
        start_time = time.time()
        # total_loss_debug = torch.zeros(5).to(device)
        log_interval = int(len(train_dataloader) / 10)

        for batch_idx, batch in enumerate(train_dataloader):
            H, H_noise, H_seq, H_pred = batch

            data, label = LoadBatch(H_noise), LoadBatch(H)

            data = data.to(device)  # (b, seq_len)
            label = label.to(device)

            output = model.train_data(data, device)
            loss = loss_fn(output[:, -15:, ...], label[:, -15:, ...])
            total_loss += loss.item()
            # total_loss_debug += loss_fn_debug(label[:, -15:, ...], output[:, -15:, ...])

            # Backpropagate the loss
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if batch_idx % 100 == 0:
                plt.figure()
                for i in range(4):
                    for j in range(2):
                        plt.subplot(4, 2, i * 2 + j + 1)
                        plt.plot(
                            label[0, :, (i + j * 4) * 2].cpu().detach().numpy(),
                            label="label",
                        )
                        plt.plot(
                            output[0, :, (i + j * 4) * 2].cpu().detach().numpy(),
                            label="output",
                        )
                        plt.xlim([75, 95])
                        plt.plot(H[0, :, j, i].cpu().numpy(), label="input")
                plt.savefig("output_train_RNN.png")
                plt.close()

            if batch_idx % log_interval == 0 and batch_idx > 0:
                lr = scheduler.optimizer.param_groups[0]["lr"]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval

                print(
                    f"| epoch {epoch:3d} | {batch_idx:5d}/{len(train_dataloader):5d} batches | "
                    f"lr {lr:e} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.4f}  ",
                    flush=True,
                )
                total_loss = 0
                start_time = time.time()

            global_step += 1

        # Run validation at the end of every epoch
        val_loss = run_validation(model, val_dataloader, device)

        elapsed = time.time() - epoch_start_time

        LossDebugString = "| Loss pred len "
        for idx, LossDebug in enumerate(val_loss):
            LossDebugString += f" {idx}: {LossDebug:5.2f}"

        print("-" * 89)
        print(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss {torch.sum(val_loss)/5} {LossDebugString}"
        )
        print("-" * 89, flush=True)

        model_filename = (
            f"{config['model_folder']}/{config['model_basename']}{epoch}.pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )

        # Log the loss to tensorboard
        writer.add_scalar("Loss/train", cur_loss, epoch)
        writer.add_scalar("Loss/val", float(sum(val_loss) / len(val_loss)), epoch)

        scheduler.step()
    return val_loss, model


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()

    model_name = f"RNN_"
    print("Model_name: ", model_name)

    config["model_folder"] = config["model_folder"] + "/RNN"
    config["experiment_name"] = (
        "/".join(config["experiment_name"].split("/", 1)) + "/RNN" + model_name
    )
    config["model_basename"] = config["model_basename"] + model_name

    loss, model = train_model(config)

    loss_list = []
    config["epoch"] = None
    for SNR in range(12, 22, 1):
        config["SNR"] = SNR
        train_dataloader, val_dataloader = get_dataset(config)
        loss = run_validation(model, val_dataloader, device)
        print(f"SNR: {SNR} | Loss: {loss}")
        loss = loss.cpu().numpy()
        loss_list.append([loss, SNR])
        print(loss_list)

    # Save loss as pickle file
    with open("loss_RNN.pkl", "wb") as f:
        pickle.dump(loss_list, f)
