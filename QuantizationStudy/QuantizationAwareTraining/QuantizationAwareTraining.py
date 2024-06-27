print("Starting training", flush=True)
import torch
from torch.utils.data import DataLoader

from pathlib import Path
import warnings
import time

import argparse
import numpy as np


from qtorch.optim import OptimLP
from qtorch.quant import fixed_point_quantize_partial, fixed_point_quantize

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import get_config

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.Informer.model import InformerStack
from Utils.dataset import SeqData, LoadBatch
from Utils.metrics import NMSELoss, NMSELossSplit


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


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
    parser.add_argument(
        "--weigth_quant_setting",
        type=str,
        help="Weight quantization setting",
        default=None,
    )
    parser.add_argument("--rounding", type=str, help="Rounding mode", default=None)
    return parser.parse_args()


def get_dataset(config):
    trainDatasetName = (
        f'../../GenerateDatasets/Datasets/{config["dataset_name"]}.pickle'
    )
    evaluateDatasetName = (
        f'../../GenerateDatasets/Datasets/{config["dataset_name"]}__validate.pickle'
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
        device,
    )

    return model.cuda() if torch.cuda.is_available() else model


def run_validation(model, val_dataloader, device):
    # Set the model to evaluation mode
    model.eval()
    # Create a metric to store the loss
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

    return loss / len(val_dataloader)


def train_model(quantizationSettings, config):
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

    # Low precision optimizer wrapper
    optimizer = OptimLP(
        optimizer,
        quantizationSettings,
        model.named_parameters(),
    )

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0

    print(f"Preloading model {config['model_filename']}")
    if torch.cuda.is_available():
        state = torch.load(config["model_filename"], map_location=device)
    else:
        state = torch.load(config["model_filename"], map_location="cpu")
    model.load_state_dict(state["model_state_dict"])

    loss_fn = NMSELoss()
    loss_fn_debug = NMSELossSplit()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        config["num_epochs"],
        T_mult=1,
        eta_min=0,
        last_epoch=-1,
        verbose=True,
    )

    for step in config["steps"]:

        for epoch in range(config["num_epochs"]):
            print(
                "Part of weights to be quantized: ",
                step,
            )
            epoch_start_time = time.time()
            torch.cuda.empty_cache()
            model.train()
            total_loss = 0.0
            total_loss_debug = torch.zeros(5).to(device)
            start_time = time.time()

            log_interval = int(len(train_dataloader) / 10)

            for batch_idx, batch in enumerate(train_dataloader):
                H, H_noise, H_seq, H_pred = batch

                data, label = LoadBatch(H_seq), LoadBatch(H_pred)

                encoder_input = data.to(device)  # (b, seq_len)
                label = label.to(device)

                decoder_input = torch.zeros_like(
                    encoder_input[:, -config["pred_len"] :, :]
                ).to(
                    device
                )  # (B, config["seq_len"])

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

                # Run the tensors through the encoder, decoder and the projection layer
                output, attn = model(
                    encoder_input,
                    range(config["seq_len"]),
                    decoder_input,
                    range(config["pred_len"] + config["label_len"]),
                )
                loss = loss_fn(output, label)
                total_loss += loss.item()

                total_loss_debug += loss_fn_debug(label, output)

                # Backpropagate the loss
                loss.backward()

                # Update the weights
                optimizer.step(percentage=step)
                optimizer.zero_grad(set_to_none=True)

                if batch_idx % log_interval == 0 and batch_idx > 0:
                    lr = scheduler.optimizer.param_groups[0]["lr"]
                    ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                    cur_loss = total_loss / log_interval

                    TotalLossDebug_values = total_loss_debug.tolist()
                    LossDebugString = "| Loss pred_len "
                    for idx, LossDebug in enumerate(TotalLossDebug_values):
                        LossDebugString += f" {idx}: {LossDebug/log_interval:5.2f}"
                    total_loss_debug = torch.zeros(5).to(device)

                    print(
                        f"| epoch {epoch:3d} | {batch_idx:5d}/{len(train_dataloader):5d} batches | "
                        f"lr {lr:e} | ms/batch {ms_per_batch:5.2f} | "
                        f"loss {cur_loss:5.2f}  " + LossDebugString,
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

            model_filename = f"weights/wl{config['wl']}_fl{config['fl']}_epoch_{epoch}_{config['rounding']}_{config['weight_quant_setting']}.pt"

            if epoch % 10 == 0 or epoch == config["num_epochs"] - 1:
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
            writer.add_scalar("Loss/val1", float(val_loss[0]), epoch)
            writer.add_scalar("Loss/val2", float(val_loss[1]), epoch)
            writer.add_scalar("Loss/val3", float(val_loss[2]), epoch)
            writer.add_scalar("Loss/val4", float(val_loss[3]), epoch)
            writer.add_scalar("Loss/val5", float(val_loss[4]), epoch)

            scheduler.step()
    return val_loss


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    config = get_config()

    args = parse_args()

    # fmt: off
    if args.d_model is not None:
        config["d_model"] = args.d_model
    if args.d_ff is not None:
        config["d_ff"] = args.d_ff
    if args.seq_len is not None:
        config["seq_len"] = args.seq_len
    if args.label_len is not None:
        config["label_len"] = args.label_len
    if args.attn is not None:
        config["attn"] = args.attn
    if args.distil is not None:
        config["distil"] = args.distil
    if args.SNR is not None:
        config["SNR"] = args.SNR
    if args.weigth_quant_setting is not None:
        config["weight_quant_setting"] = args.weigth_quant_setting
    if args.rounding is not None:
        config["rounding"] = args.rounding
        if args.weigth_quant_setting == "partial":
            if config["rounding"] == "proximal":
                config["steps"] = [0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.98,0.99, 0.995,0.998,0.999,0.9995,0.9998,0.9999,1.0]
            elif config["rounding"] == "stochastic":
                config["steps"] = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5, 0.55,0.60,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
            elif config["rounding"] == "distant":
                config["steps"] = [0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.15, 0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
            config["num_epochs"] = 10
        else:
            config["steps"] = [1.0]
            config["num_epochs"] = 70

    # fmt:on
    config["model_filename"] = "Weights/tmodel_pretrained.pt"
    config["experiment_name"] = (
        f"runs/{config['weight_quant_setting']}/{config['rounding']}"
    )
    loss_list = []
    for wl in [2]:
        fl = wl - 4
        config["wl"] = wl
        config["fl"] = fl
        print(f"Quantizing with wl: {wl}, fl: {fl}")

        if config["weight_quant_setting"] == "partial":
            weight_quant = [
                lambda x, percentage: fixed_point_quantize_partial(
                    x, wl=wl, fl=fl, percentage=percentage, rounding=config["rounding"]
                ),
                "partial",
            ]
        else:
            weight_quant = [
                lambda x, percentage: fixed_point_quantize(
                    x, wl=wl, fl=fl, rounding=config["rounding"]
                ),
                "full",
            ]

        # fmt: off
        quantizationSettings = {
        # Encoders
        # Encoder 0 attn 0
        "encoder.encoders.0.attn_layers.0.attention.query_projection.weight": { "weight_quant": weight_quant[0] },
        "encoder.encoders.0.attn_layers.0.attention.key_projection.weight"  : { "weight_quant": weight_quant[0]},
        "encoder.encoders.0.attn_layers.0.attention.value_projection.weight": { "weight_quant": weight_quant[0]},
        "encoder.encoders.0.attn_layers.0.attention.out_projection.weight"  : { "weight_quant": weight_quant[0]},
        "encoder.encoders.0.attn_layers.0.conv1.weight"                    : { "weight_quant": weight_quant[0]},
        "encoder.encoders.0.attn_layers.0.conv2.weight"                    : { "weight_quant": weight_quant[0]},

        # Encoder 0 attn 1
        "encoder.encoders.0.attn_layers.1.attention.query_projection.weight": { "weight_quant": weight_quant[0]},
        "encoder.encoders.0.attn_layers.1.attention.key_projection.weight"  : { "weight_quant": weight_quant[0]},
        "encoder.encoders.0.attn_layers.1.attention.value_projection.weight": { "weight_quant": weight_quant[0]},
        "encoder.encoders.0.attn_layers.1.attention.out_projection.weight"  : { "weight_quant": weight_quant[0]},
        "encoder.encoders.0.attn_layers.1.conv1.weight"                    : { "weight_quant": weight_quant[0]},
        "encoder.encoders.0.attn_layers.1.conv2.weight"                    : { "weight_quant": weight_quant[0]},
        
        # Encoder 0 attn 2
        "encoder.encoders.0.attn_layers.2.attention.query_projection.weight": { "weight_quant": weight_quant[0]},
        "encoder.encoders.0.attn_layers.2.attention.key_projection.weight"  : { "weight_quant": weight_quant[0]},
        "encoder.encoders.0.attn_layers.2.attention.value_projection.weight": { "weight_quant": weight_quant[0]},
        "encoder.encoders.0.attn_layers.2.attention.out_projection.weight"  : { "weight_quant": weight_quant[0]},
        "encoder.encoders.0.attn_layers.2.conv1.weight"                    : { "weight_quant": weight_quant[0]},
        "encoder.encoders.0.attn_layers.2.conv2.weight"                    : { "weight_quant": weight_quant[0]},
        
        # Encoder 0 attn 3
        "encoder.encoders.0.attn_layers.3.attention.query_projection.weight": { "weight_quant": weight_quant[0]},
        "encoder.encoders.0.attn_layers.3.attention.key_projection.weight"  : { "weight_quant": weight_quant[0]},
        "encoder.encoders.0.attn_layers.3.attention.value_projection.weight": { "weight_quant": weight_quant[0]},
        "encoder.encoders.0.attn_layers.3.attention.out_projection.weight"  : { "weight_quant": weight_quant[0]},
        "encoder.encoders.0.attn_layers.3.conv1.weight"                    : { "weight_quant": weight_quant[0]},
        "encoder.encoders.0.attn_layers.3.conv2.weight"                    : { "weight_quant": weight_quant[0]},
        
        # Encoder 1 attn 0
        "encoder.encoders.1.attn_layers.0.attention.query_projection.weight": { "weight_quant": weight_quant[0]},
        "encoder.encoders.1.attn_layers.0.attention.key_projection.weight"  : { "weight_quant": weight_quant[0]},
        "encoder.encoders.1.attn_layers.0.attention.value_projection.weight": { "weight_quant": weight_quant[0]},
        "encoder.encoders.1.attn_layers.0.attention.out_projection.weight"  : { "weight_quant": weight_quant[0]},
        "encoder.encoders.1.attn_layers.0.conv1.weight"                    : { "weight_quant": weight_quant[0]},
        "encoder.encoders.1.attn_layers.0.conv2.weight"                    : { "weight_quant": weight_quant[0]},
        
        # Encoder 1 attn 1
        "encoder.encoders.1.attn_layers.1.attention.query_projection.weight": { "weight_quant": weight_quant[0]},
        "encoder.encoders.1.attn_layers.1.attention.key_projection.weight"  : { "weight_quant": weight_quant[0]},
        "encoder.encoders.1.attn_layers.1.attention.value_projection.weight": { "weight_quant": weight_quant[0]},
        "encoder.encoders.1.attn_layers.1.attention.out_projection.weight"  : { "weight_quant": weight_quant[0]},
        "encoder.encoders.1.attn_layers.1.conv1.weight"                    : { "weight_quant": weight_quant[0]},
        "encoder.encoders.1.attn_layers.1.conv2.weight"                    : { "weight_quant": weight_quant[0]},
        
        # Encoder 1 attn 2
        "encoder.encoders.1.attn_layers.2.attention.query_projection.weight": { "weight_quant": weight_quant[0]},
        "encoder.encoders.1.attn_layers.2.attention.key_projection.weight"  : { "weight_quant": weight_quant[0]},
        "encoder.encoders.1.attn_layers.2.attention.value_projection.weight": { "weight_quant": weight_quant[0]},
        "encoder.encoders.1.attn_layers.2.attention.out_projection.weight"  : { "weight_quant": weight_quant[0]},
        "encoder.encoders.1.attn_layers.2.conv1.weight"                    : { "weight_quant": weight_quant[0]},
        "encoder.encoders.1.attn_layers.2.conv2.weight"                    : { "weight_quant": weight_quant[0]},
        
        # Decoder
        # Decoder 0 attn 0
        "decoder.layers.0.self_attention.query_projection.weight": { "weight_quant": weight_quant[0]},
        "decoder.layers.0.self_attention.key_projection.weight"  : { "weight_quant": weight_quant[0]},
        "decoder.layers.0.self_attention.value_projection.weight": { "weight_quant": weight_quant[0]},
        "decoder.layers.0.self_attention.out_projection.weight"  : { "weight_quant": weight_quant[0]},
        "decoder.layers.0.cross_attention.query_projection.weight": { "weight_quant": weight_quant[0]},
        "decoder.layers.0.cross_attention.key_projection.weight"  : { "weight_quant": weight_quant[0]},
        "decoder.layers.0.cross_attention.value_projection.weight": { "weight_quant": weight_quant[0]},
        "decoder.layers.0.cross_attention.out_projection.weight"  : { "weight_quant": weight_quant[0]},
        "decoder.layers.0.conv1.weight"                          : { "weight_quant": weight_quant[0]},
        "decoder.layers.0.conv2.weight"                          : { "weight_quant": weight_quant[0]},
        
        # Decoder 1 attn 0
        "decoder.layers.1.self_attention.query_projection.weight": { "weight_quant": weight_quant[0]},
        "decoder.layers.1.self_attention.key_projection.weight"  : { "weight_quant": weight_quant[0]},
        "decoder.layers.1.self_attention.value_projection.weight": { "weight_quant": weight_quant[0]},
        "decoder.layers.1.self_attention.out_projection.weight"  : { "weight_quant": weight_quant[0]},
        "decoder.layers.1.cross_attention.query_projection.weight": { "weight_quant": weight_quant[0]},
        "decoder.layers.1.cross_attention.key_projection.weight"  : { "weight_quant": weight_quant[0]},
        "decoder.layers.1.cross_attention.value_projection.weight": { "weight_quant": weight_quant[0]},
        "decoder.layers.1.cross_attention.out_projection.weight"  : { "weight_quant": weight_quant[0]},
        "decoder.layers.1.conv1.weight"                          : { "weight_quant": weight_quant[0]},
        "decoder.layers.1.conv2.weight"                          : { "weight_quant": weight_quant[0]},
        
        # Decoder 2 attn 0
        "decoder.layers.2.self_attention.query_projection.weight": { "weight_quant": weight_quant[0]},
        "decoder.layers.2.self_attention.key_projection.weight"  : { "weight_quant": weight_quant[0]},
        "decoder.layers.2.self_attention.value_projection.weight": { "weight_quant": weight_quant[0]},
        "decoder.layers.2.self_attention.out_projection.weight"  : { "weight_quant": weight_quant[0]},
        "decoder.layers.2.cross_attention.query_projection.weight": { "weight_quant": weight_quant[0]},
        "decoder.layers.2.cross_attention.key_projection.weight"  : { "weight_quant": weight_quant[0]},
        "decoder.layers.2.cross_attention.value_projection.weight": { "weight_quant": weight_quant[0]},   
        "decoder.layers.2.cross_attention.out_projection.weight"  : { "weight_quant": weight_quant[0]},
        "decoder.layers.2.conv1.weight"                          : { "weight_quant": weight_quant[0]},
        "decoder.layers.2.conv2.weight"                          : { "weight_quant": weight_quant[0]},
        }
        # fmt:on
        print(f"Quantizing with {weight_quant[1]} weights")
        print(f"Quantizing with {config['rounding']} rounding mode")
        print(f"Quantizing with wl: {wl}, fl: {fl}")

        loss = train_model(quantizationSettings, config)
        data_np = [[tensor.cpu().numpy()] for tensor in loss]

        loss_list.append([data_np, wl])

        # Save the loss list to a file
        import pickle

        # Save the loss list to a file using pickle
        with open(
            f"loss_list_{config['weight_quant_setting']}_{config['rounding']}.pickle",
            "wb",
        ) as f:
            pickle.dump(loss_list, f)
