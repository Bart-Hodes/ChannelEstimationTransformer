import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from pathlib import Path
import warnings
import time
from datetime import datetime

from TransformerModel.model import build_transformer
from dataset import SeqData, LoadBatch
from config import latest_weights_file_path, get_weights_file_path, get_config
from metrics import NMSELoss, NMSELossSplit

from torch.utils.tensorboard import SummaryWriter


def get_dataset(config):
    trainDatasetName = f'../GenerateDatasets/Datasets/{config["dataset_name"]}.pickle'
    evaluateDatasetName = (
        f'../GenerateDatasets/Datasets/{config["dataset_name"]}__validate.pickle'
    )

    print(f"Sliding window: {config['sliding_window']}")
    trainData = SeqData(
        trainDatasetName,
        config["seq_len"],
        config["pred_len"],
        config["sliding_window"],
        SNR=config["SNR"],
    )
    trainLoader = DataLoader(
        dataset=trainData,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    evaluateData = SeqData(
        evaluateDatasetName,
        config["seq_len"],
        config["pred_len"],
        config["sliding_window"],
        SNR=config["SNR"],
    )
    evaluaterLoader = DataLoader(
        dataset=evaluateData,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    return trainLoader, evaluaterLoader


def get_model(config, seq_len, pred_len):
    model = build_transformer(
        config["enc_in"],
        config["dec_in"],
        config["seq_len"],
        config["pred_len"],
        config["label_len"],
        config["d_model"],
        config["d_layers"],
        config["n_heads"],
        config["dropout"],
        config["d_ff"],
    )
    return model.cuda() if torch.cuda.is_available() else model


Debug = True


def run_validation(model, val_dataloader, device):
    # Set the model to evaluation mode
    model.eval()
    # Create a metric to store the loss
    loss_fn = NMSELossSplit()
    loss = torch.zeros(5).to(device)
    # Iterate over the validation dataloader
    for batch in val_dataloader:
        H, H_noise, H_seq, H_pred = batch
        first_inference = True
        for seq_index in range(config["sliding_window"]):
            if Debug:
                data, label = LoadBatch(H_seq), LoadBatch(H_pred)

                label = label.to(device)
                encoder_input = data.to(device)  # (b, seq_len)

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
                # Run the tensors through the encoder, decoder and the projection layer
                output = model(
                    encoder_input,
                    decoder_input,
                )

                # Save plot
                output = output.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                plt.figure()
                for j in range(4):
                    for i in range(2):
                        plt.subplot(4, 2, j * 2 + i + 1)
                        plt.plot(output[0, :, j * 2 + i])
                        plt.plot(label[0, :, j * 2 + i])
                plt.xlabel("Output")
                plt.ylabel("Label")
                plt.title("Output vs Label")
                plt.savefig("output_vs_labe2.png")
                plt.close()

                ############################################################################

            H_seq = H_noise[:, seq_index : seq_index + config["seq_len"], :]
            H_pred = H[
                :,
                seq_index
                + config["seq_len"] : seq_index
                + config["seq_len"]
                + config["pred_len"],
                :,
            ]

            data, label = LoadBatch(H_seq), LoadBatch(H_pred)

            label = label.to(device)
            encoder_input = data.to(device)
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

            with torch.no_grad():
                output = model(
                    encoder_input,
                    decoder_input,
                )
                loss += loss_fn(output, label)

            if Debug:
                # print(loss)

                # Save plot
                output = output.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                plt.figure()
                for j in range(4):
                    for i in range(2):
                        plt.subplot(4, 2, j * 2 + i + 1)
                        plt.plot(output[0, :, j * 2 + i])
                        plt.plot(label[0, :, j * 2 + i])
                plt.xlabel("Output")
                plt.ylabel("Label")
                plt.title("Output vs Label")
                plt.savefig("output_vs_label.png")
                plt.close()

                quit()

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

    model = get_model(config, config["seq_len"], config["pred_len"])
    # Tensorboard
    writer = SummaryWriter(
        config["experiment_name"] + datetime.now().strftime("%Y%m%d%H%M%S")
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config["num_epochs"], eta_min=0, last_epoch=-1
    )

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config["preload"]
    model_filename = (
        latest_weights_file_path(config)
        if preload == "latest"
        else get_weights_file_path(config, preload) if preload else None
    )
    if model_filename:
        print(f"Preloading model {model_filename}")
        if torch.cuda.is_available():
            state = torch.load(model_filename, map_location=device)
        else:
            state = torch.load(model_filename, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("No model to preload, starting from scratch")

    loss_fn = NMSELoss()
    loss_fn_debug = NMSELossSplit()

    val_loss = run_validation(model, val_dataloader, device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        epoch_start_time = time.time()
        torch.cuda.empty_cache()
        model.train()
        total_loss = 0.0
        start_time = time.time()
        total_loss_debug = torch.zeros(5).to(device)
        log_interval = int(len(train_dataloader) / 10)

        for batch_idx, batch in enumerate(train_dataloader):
            H, H_noise, H_seq, H_pred = batch
            first_inference = True

            data, label = LoadBatch(H_seq), LoadBatch(H_pred)

            label = label.to(device)
            encoder_input = data.to(device)  # (b, seq_len)

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
            # Run the tensors through the encoder, decoder and the projection layer
            output = model(
                encoder_input,
                decoder_input,
            )

            loss = loss_fn(output, label)
            total_loss += loss.item()
            total_loss_debug += loss_fn_debug(label, output)

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

        LossDebugString = "| Loss pred len "
        for idx, LossDebug in enumerate(val_loss):
            LossDebugString += f" {idx}: {LossDebug:5.2f}"

        print("-" * 89)
        print(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss {torch.sum(val_loss)/5} {LossDebugString}"
        )
        print("-" * 89, flush=True)

        # Save the model at the end of every epoch
        if epoch % 10 == 0 or epoch == config["num_epochs"] - 1:
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
