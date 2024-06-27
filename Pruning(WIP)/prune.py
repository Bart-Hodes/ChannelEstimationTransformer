import torch
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import torch.optim as optim
import matplotlib.pyplot as plt

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
    model.eval()
    loss_fn = NMSELossSplit()
    loss = torch.zeros(5).to(device)

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

    plt.figure()
    x = np.arange(20, 25, 1)
    for i in range(4):
        for j in range(2):
            plt.subplot(4, 2, i * 2 + j + 1)
            plt.plot(x, label[0, :, (i + j * 4) * 2].cpu().numpy(), label="label")
            plt.plot(x, output[0, :, (i + j * 4) * 2].cpu().numpy(), label="output")
            plt.plot(H[0, -25:, j, i].cpu().numpy(), label="input")
    plt.savefig("output.png")
    plt.close()
    return loss / len(val_dataloader)


def train_model(config):
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

    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader = get_dataset(config)

    model = get_model(config)

    writer = SummaryWriter(config["experiment_name"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config["num_epochs"], eta_min=0, last_epoch=-1
    )

    initial_epoch = 0
    global_step = 0

    model_filename = (
        f"{config['model_folder']}/{config['model_basename']}{config['preload']}.pt"
    )
    config["preload"] = "Weights/tmodel_pretrained"
    model_filename = "Weights/tmodel_pretrained.pt"
    if config["preload"] is not None:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        global_step = state["global_step"]

    from neural_compressor.pruner.pruning import Pruning, WeightPruningConfig

    pruner_config = WeightPruningConfig(config)
    pruner = Pruning(pruner_config)  # Define a pruning object
    pruner.model = model  # Set model object to prune
    pruner.on_train_begin()

    loss_fn = NMSELoss()
    loss_fn_debug = NMSELossSplit()

    for epoch in range(config["num_epochs"]):
        epoch_start_time = time.time()
        torch.cuda.empty_cache()
        model.train()
        total_loss = 0.0
        start_time = time.time()
        total_loss_debug = torch.zeros(5).to(device)
        log_interval = int(len(train_dataloader) / 10)

        for batch_idx, batch in enumerate(train_dataloader):
            pruner.on_step_begin(batch_idx)
            H, H_noise, H_seq, H_pred = batch

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

            total_loss_debug += loss_fn_debug(label, output)

            loss.backward()
            pruner.on_before_optimizer_step()
            optimizer.step()
            pruner.on_after_optimizer_step()
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

        if epoch % 100 == 0 or epoch == config["num_epochs"] - 1:
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

        scheduler.step()
        writer.add_scalar("Loss/train", cur_loss, epoch)
        writer.add_scalar("Loss/val", float(sum(val_loss) / len(val_loss)), epoch)

    return val_loss, model


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()

    config.update(
        {
            "target_sparsity": 0.9,
            "pruning_type": "snip_momentum",
            "pattern": "4x1",
            "excluded_op_names": [
                "embedding.*",
                "classifier.*",
            ],
        }
    )

    model_name = f"ei_{config['enc_in']}_di_{config['dec_in']}_co_{config['c_out']}_sl_{config['seq_len']}_ll_{config['label_len']}_pl_{config['pred_len']}_f_{config['factor']}_dm_{config['d_model']}_nh_{config['n_heads']}_el_{config['e_layers']}_dl_{config['d_layers']}_df_{config['d_ff']}_do_{config['dropout']}_at_{config['attn']}_em_{config['embed']}_ac_{config['activation']}"
    print("Model_name: ", model_name)

    config["model_folder"] = config["model_folder"] + "/Informer"
    config["experiment_name"] = (
        "/".join(config["experiment_name"].split("/", 1)) + "/Informer" + model_name
    )
    config["model_basename"] = config["model_basename"] + model_name

    loss, model = train_model(config)

    loss_np = loss.cpu().numpy()

    with open("loss.pkl", "wb") as f:
        pickle.dump(loss_np, f)
