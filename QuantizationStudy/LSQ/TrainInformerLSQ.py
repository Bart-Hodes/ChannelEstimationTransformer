import argparse
import json
import time
import warnings
from pathlib import Path
from json import JSONEncoder

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from config import get_config

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.InformerLSQ.LSQ import Conv1dLSQ, LinearLSQ
from models.InformerLSQ.model import InformerStack
from Utils.dataset import LoadBatch, SeqData
from Utils.metrics import NMSELoss, NMSELossSplit


class EncodeTensor(JSONEncoder, Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super().default(obj)


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
    train_dataset_path = (
        f'../../GenerateDatasets/Datasets/{config["dataset_name"]}.pickle'
    )
    validate_dataset_path = (
        f'../../GenerateDatasets/Datasets/{config["dataset_name"]}__validate.pickle'
    )

    train_data = SeqData(
        train_dataset_path, config["seq_len"], config["pred_len"], SNR=config["SNR"]
    )
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    validate_data = SeqData(
        validate_dataset_path, config["seq_len"], config["pred_len"], SNR=config["SNR"]
    )
    validate_loader = DataLoader(
        dataset=validate_data,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    return train_loader, validate_loader


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


def run_validation(model, val_dataloader, device, config):
    model.eval()
    loss_fn = NMSELossSplit()
    loss = torch.zeros(5).to(device)

    for H, H_noise, H_seq, H_pred in val_dataloader:
        data, label = LoadBatch(H_seq), LoadBatch(H_pred)
        encoder_input = data.to(device)
        label = label.to(device)
        decoder_input = torch.zeros_like(encoder_input[:, -config["pred_len"] :, :]).to(
            device
        )
        decoder_input = torch.cat(
            [
                encoder_input[
                    :,
                    config["seq_len"] - config["label_len"] : config["seq_len"],
                    :,
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

    return loss / len(val_dataloader)


def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(
            f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3} GB"
        )
    elif device == "mps":
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")

    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader = get_dataset(config)
    model = get_model(config)
    writer = SummaryWriter(config["experiment_name"] + "_" + str(config["num_bits"]))

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config["num_epochs"], eta_min=0, last_epoch=-1
    )

    model_filename = "Weights/tmodel_pretrained.pt"
    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(
            model_filename,
            map_location=config["device"] if torch.cuda.is_available() else "cpu",
        )
        model.load_state_dict(state["model_state_dict"], strict=False)

    # Set initial values for stepsize
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

    loss_fn = NMSELoss()
    loss_fn_debug = NMSELossSplit()

    for epoch in range(config["num_epochs"]):
        epoch_start_time = time.time()
        torch.cuda.empty_cache()
        model.train()
        total_loss = 0.0
        total_loss_debug = torch.zeros(5).to(device)
        start_time = time.time()
        log_interval = int(len(train_dataloader) / 10)

        for batch_idx, (H, H_noise, H_seq, H_pred) in enumerate(train_dataloader):
            data, label = LoadBatch(H_seq), LoadBatch(H_pred)
            encoder_input = data.to(device)
            label = label.to(device)
            decoder_input = torch.zeros_like(
                encoder_input[:, -config["pred_len"] :, :]
            ).to(device)
            decoder_input = torch.cat(
                [
                    encoder_input[
                        :,
                        config["seq_len"] - config["label_len"] : config["seq_len"],
                        :,
                    ],
                    decoder_input,
                ],
                dim=1,
            )

            output, attn = model(
                encoder_input,
                range(config["seq_len"]),
                decoder_input,
                range(config["pred_len"] + config["label_len"]),
            )
            loss = loss_fn(output, label)
            total_loss += loss.item()

            loss_debug = loss_fn_debug(output, label)
            total_loss_debug += loss_debug

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if batch_idx % log_interval == 0 and batch_idx > 0:
                lr = scheduler.optimizer.param_groups[0]["lr"]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval

                loss_debug_values = total_loss_debug.tolist()
                loss_debug_str = "| Loss pred_len " + " ".join(
                    [
                        f"{i}: {v/log_interval:5.2f}"
                        for i, v in enumerate(loss_debug_values)
                    ]
                )
                total_loss_debug = torch.zeros(5).to(device)

                print(
                    f"| epoch {epoch:3d} | {batch_idx:5d}/{len(train_dataloader):5d} batches | lr {lr:e} | ms/batch {ms_per_batch:5.2f} | loss {cur_loss:5.4f}  {loss_debug_str}",
                    flush=True,
                )
                total_loss = 0
                start_time = time.time()

        val_loss = run_validation(model, val_dataloader, device, config)
        elapsed = time.time() - epoch_start_time

        print("-" * 89)
        print(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss {torch.sum(val_loss)/5:5.6f}"
        )
        print("-" * 89, flush=True)

        if epoch % 10 == 0 or epoch == config["num_epochs"] - 1:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": epoch * len(train_dataloader),
                },
                f"weights/nbits{config['num_bits']}_epoch_{epoch}.pt",
            )

        # Log the loss to tensorboard
        writer.add_scalar("Loss/train", cur_loss, epoch)
        writer.add_scalar("Loss/val", float(sum(val_loss) / len(val_loss)), epoch)
        writer.add_scalar("Loss/val1", float(val_loss[0]), epoch)
        writer.add_scalar("Loss/val2", float(val_loss[1]), epoch)
        writer.add_scalar("Loss/val3", float(val_loss[2]), epoch)
        writer.add_scalar("Loss/val4", float(val_loss[3]), epoch)
        writer.add_scalar("Loss/val5", float(val_loss[4]), epoch)
        # testing

        scheduler.step()
    return val_loss


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()

    config.update(
        {
            "enc_in": 16,
            "dec_in": 16,
            "c_out": 16,
            "seq_len": 90,
            "label_len": 10,
            "pred_len": 5,
            "factor": 5,
            "d_model": 128,
            "n_heads": 8,
            "e_layers": [4, 3],
            "d_layers": 3,
            "d_ff": 64,
            "dropout": 0.05,
            "attn": "full",
            "embed": "fixed",
            "activation": "gelu",
            "output_attention": False,
            "distil": True,
            "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        }
    )

    e_layers_str = "_".join(map(str, config["e_layers"]))
    # config["model_folder"] = (
    #     f"{config['model_path']}/{config['experiment_name']}_{config['d_model']}_{config['d_ff']}_{e_layers_str}_{config['attn']}_{config['distil']}_nbits{config['num_bits']}"
    # )

    for config["num_bits"] in range(8, 12):
        val_loss = train_model(config)
        print(f"Val loss: {val_loss}")
