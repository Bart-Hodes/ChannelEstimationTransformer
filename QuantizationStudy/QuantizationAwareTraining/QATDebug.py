import torch
from torch.utils.data import DataLoader

from pathlib import Path
import warnings
import time

import matplotlib.pyplot as plt

import argparse
import numpy as np

from InformerModel.model import InformerStack
from dataset import SeqData, LoadBatch
from config import get_config
from metrics import NMSELoss, NMSELossSplit

from torch.utils.tensorboard import SummaryWriter

from qtorch.optim import OptimLP
from qtorch.quant import fixed_point_quantize_partial, fixed_point_quantize

import numpy as np

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

    plt.figure()
    x = np.arange(20, 25, 1)
    print(H.shape)
    print(label.shape)
    print(output.shape)
    for i in range(4):
        for j in range(2):
            plt.subplot(4, 2, i * 2 + j + 1)
            plt.plot(x, label[0, :, (i + j * 4) * 2].cpu().numpy(), label="label")
            plt.plot(x, output[0, :, (i + j * 4) * 2].cpu().numpy(), label="output")
            plt.plot(H[0, -25:, j, i].cpu().numpy(), label="input")
    plt.savefig("output.png")
    plt.close()
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config["num_epochs"], eta_min=0, last_epoch=-1
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
            optimizer.step(percentage=1)
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

        model_filename = f"weights/wl{config['wl']}_fl{config['fl']}_epoch_{epoch}.pt"
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

    return val_loss


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()

    # config["model_folder"] = config["model_folder"] + "/InformerQAT"
    # config["experiment_name"] = (
    #     "/".join(config["experiment_name"].split("/", 1)) + "/InformerQAT" + model_name
    # )
    # config["model_basename"] = config["model_basename"] + model_name
    # print("experiment_name: ", config["experiment_name"])
    # print("Dataset Name: ", config["dataset_name"])

    config["num_epochs"] = 20
    config["model_filename"] = "Weights/tmodel_pretrained.pt"

    config["rounding"] = "stochastic"
    wl = 4
    fl = wl - 1

    # weight_quant_partial = [
    #     lambda x, percentage: fixed_point_quantize_partial(
    #         x, wl=wl, fl=fl, percentage=percentage, rounding=rounding
    #     ),
    #     "partial",
    # ]

    weight_quant_full = [
        lambda x, percentage: fixed_point_quantize(
            x, wl=wl, fl=fl, rounding=config["rounding"]
        ),
        "full",
    ]

    weight_quant = weight_quant_full

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
