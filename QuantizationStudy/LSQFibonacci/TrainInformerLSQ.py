import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from pathlib import Path
import warnings
import time

from config import get_config

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.InformerLSQFibbinary.model import InformerStack
from models.InformerLSQFibbinary.LSQ import LinearLSQ, Conv1dLSQ
from Utils.dataset import SeqData, LoadBatch
from Utils.metrics import NMSELoss, NMSELossSplit

from torch.utils.tensorboard import SummaryWriter


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
    writer = SummaryWriter(config["experiment_name"] + "_" + str(config["num_bits"]))

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config["num_epochs"], eta_min=0, last_epoch=-1
    )
    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    # preload = config["preload"]
    # model_filename = (
    #     latest_weights_file_path(config)
    #     if preload == "latest"
    #     else get_weights_file_path(config, preload) if preload else None
    # )

    model_filename = "Weights/tmodel_pretrained.pt"
    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(
            model_filename,
            map_location=config["device"] if torch.cuda.is_available() else "cpu",
        )
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        print("No model to preload, starting from scratch")

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

        for batch_idx, batch in enumerate(train_dataloader):

            H, H_noise, H_seq, H_pred = batch

            data, label = LoadBatch(H_seq), LoadBatch(H_pred)

            encoder_input = data.to(device)  # (b, seq_len)
            label = label.to(device)

            decoder_input = torch.zeros_like(
                encoder_input[:, -config["pred_len"] :, :]
            ).to(
                device
            )  # (B, seq_len)
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

            # # Run the tensors through the encoder, decoder and the projection layer
            output, attn = model(
                encoder_input,
                range(config["seq_len"]),
                decoder_input,
                range(config["pred_len"] + config["label_len"]),
            )
            loss = loss_fn(output, label)
            total_loss += loss.item()

            # Calculate the loss for each prediction step
            loss_debug = loss_fn_debug(output, label)
            total_loss_debug += loss_debug

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
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
                    f"loss {cur_loss:5.4f}  " + LossDebugString,
                    flush=True,
                )
                total_loss = 0
                start_time = time.time()

            global_step += 1

        # Run validation at the end of every epoch
        val_loss = run_validation(model, val_dataloader, device)

        elapsed = time.time() - epoch_start_time

        print("-" * 89)
        print(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss {torch.sum(val_loss)/5:5.6f}"
        )
        print("-" * 89, flush=True)

        if epoch % 10 == 0 or epoch == config["num_epochs"] - 1:
            model_filename = f"weights/nbits{config['num_bits']}_epoch_{epoch}.pt"
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


import sys
from cProfile import Profile

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()

    print(len(sys.argv))

    # Add network settings to the config
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

    loss_list = []
    for nbits in range(2, 12):
        config["num_bits"] = nbits
        print(f"Training with {nbits} bits")

        loss = train_model(config)

        data_np = [[tensor.cpu().numpy()] for tensor in loss]

        loss_list.append([data_np, nbits])

    # Save the loss list to a file
    import pickle

    print("Loss list: ", loss_list)

    # Save the loss list to a file using pickle
    with open(f"loss_list.pickle", "wb") as f:
        pickle.dump(loss_list, f)
