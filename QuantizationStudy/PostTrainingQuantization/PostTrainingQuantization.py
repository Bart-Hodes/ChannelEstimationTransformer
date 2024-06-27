import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from config import get_config

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.Informer.model import InformerStack
from Utils.dataset import SeqData, LoadBatch
from Utils.metrics import NMSELoss, NMSELossSplit

from qtorch.optim import OptimLP
from qtorch.quant import fixed_point_quantize

import numpy as np
import qtorch
import pickle

# Network settings
enc_in = 16
dec_in = 16
c_out = 16
seq_len = 90
label_len = 10
pred_len = 5
factor = 5
d_model = 128
n_heads = 8
e_layers = [4, 3]
d_layers = 3
d_ff = 64
dropout = 0.05
attn = "full"
embed = "fixed"
activation = "gelu"
output_attention = False
distil = True
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Remove unwanted characters and replace them with underscores
e_layers_str = "_".join(map(str, e_layers))
attn = attn.replace(" ", "_")
embed = embed.replace(" ", "_")
activation = activation.replace(" ", "_")

model_name = f"{enc_in}_{dec_in}_{c_out}_{seq_len}_{label_len}_{pred_len}_{factor}_{d_model}_{n_heads}_{e_layers_str}_{d_layers}_{d_ff}_{dropout}_{attn}_{embed}_{activation}"
print("Model_name: ", model_name)


def get_dataset(config):
    evaluateDatasetName = (
        f'../GenerateDatasets/Datasets/{config["dataset_name"]}__validate.pickle'
    )

    evaluateData = SeqData(evaluateDatasetName, seq_len, pred_len, SNR=21)
    evaluaterLoader = DataLoader(
        dataset=evaluateData,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )
    return evaluaterLoader


def get_model():

    model = InformerStack(
        enc_in,
        dec_in,
        c_out,
        seq_len,
        label_len,
        pred_len,
        factor,
        d_model,
        n_heads,
        e_layers,
        d_layers,
        d_ff,
        dropout,
        attn,
        embed,
        activation,
        output_attention,
        distil,
        device,
    )
    model = model.cuda() if torch.cuda.is_available() else model
    return (
        model,
        f"ei_{enc_in}_di_{dec_in}_co_{c_out}_sl_{seq_len}_ll_{label_len}_pl_{pred_len}_f_{factor}_dm_{d_model}_nh_{n_heads}_el_{e_layers}_dl_{d_layers}_df_{d_ff}_do_{dropout}_at_{attn}_em_{embed}_ac_{activation}",
    )


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
        decoder_input = torch.zeros_like(encoder_input[:, -pred_len:, :]).to(device)
        decoder_input = torch.cat(
            [
                encoder_input[:, seq_len - label_len : seq_len, :],
                decoder_input,
            ],
            dim=1,
        )
        with torch.no_grad():
            output, attn = model(
                encoder_input,
                range(seq_len),
                decoder_input,
                range(pred_len + label_len),
            )
            loss += loss_fn(output, label)

    val_loss = loss / len(val_dataloader)
    return val_loss


if __name__ == "__main__":
    config = get_config()
    model, model_name = get_model()

    model_filename_pretrained = "Weights/tmodel_pretrained.pt"

    for rounding in ["stochastic", "nearest"]:
        loss_list = []
        for wl in range(4, 16):

            fl = wl - 4
            weight_quant = lambda x: qtorch.quant.fixed_point_quantize(
                x, wl=wl, fl=fl, rounding=rounding
            )
            print(f"Quantizing with wl: {wl}, fl: {fl}", flush=True)

            # fmt: off
            quantizationSettings = {
            # Encoders
            # Encoder 0 attn 0
            "encoder.encoders.0.attn_layers.0.attention.query_projection.weight": { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.0.attention.key_projection.weight"  : { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.0.attention.value_projection.weight": { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.0.attention.out_projection.weight"  : { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.0.conv1.weight"                    : { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.0.conv2.weight"                    : { "weight_quant": weight_quant },

            # Encoder 0 attn 1
            "encoder.encoders.0.attn_layers.1.attention.query_projection.weight": { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.1.attention.key_projection.weight"  : { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.1.attention.value_projection.weight": { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.1.attention.out_projection.weight"  : { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.1.conv1.weight"                    : { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.1.conv2.weight"                    : { "weight_quant": weight_quant },
            
            # Encoder 0 attn 2
            "encoder.encoders.0.attn_layers.2.attention.query_projection.weight": { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.2.attention.key_projection.weight"  : { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.2.attention.value_projection.weight": { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.2.attention.out_projection.weight"  : { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.2.conv1.weight"                    : { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.2.conv2.weight"                    : { "weight_quant": weight_quant },
            
            # Encoder 0 attn 3
            "encoder.encoders.0.attn_layers.3.attention.query_projection.weight": { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.3.attention.key_projection.weight"  : { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.3.attention.value_projection.weight": { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.3.attention.out_projection.weight"  : { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.3.conv1.weight"                    : { "weight_quant": weight_quant },
            "encoder.encoders.0.attn_layers.3.conv2.weight"                    : { "weight_quant": weight_quant },
            
            # Encoder 1 attn 0
            "encoder.encoders.1.attn_layers.0.attention.query_projection.weight": { "weight_quant": weight_quant },
            "encoder.encoders.1.attn_layers.0.attention.key_projection.weight"  : { "weight_quant": weight_quant },
            "encoder.encoders.1.attn_layers.0.attention.value_projection.weight": { "weight_quant": weight_quant },
            "encoder.encoders.1.attn_layers.0.attention.out_projection.weight"  : { "weight_quant": weight_quant },
            "encoder.encoders.1.attn_layers.0.conv1.weight"                    : { "weight_quant": weight_quant },
            "encoder.encoders.1.attn_layers.0.conv2.weight"                    : { "weight_quant": weight_quant },
            
            # Encoder 1 attn 1
            "encoder.encoders.1.attn_layers.1.attention.query_projection.weight": { "weight_quant": weight_quant },
            "encoder.encoders.1.attn_layers.1.attention.key_projection.weight"  : { "weight_quant": weight_quant },
            "encoder.encoders.1.attn_layers.1.attention.value_projection.weight": { "weight_quant": weight_quant },
            "encoder.encoders.1.attn_layers.1.attention.out_projection.weight"  : { "weight_quant": weight_quant },
            "encoder.encoders.1.attn_layers.1.conv1.weight"                    : { "weight_quant": weight_quant },
            "encoder.encoders.1.attn_layers.1.conv2.weight"                    : { "weight_quant": weight_quant },
            
            # Encoder 1 attn 2
            "encoder.encoders.1.attn_layers.2.attention.query_projection.weight": { "weight_quant": weight_quant },
            "encoder.encoders.1.attn_layers.2.attention.key_projection.weight"  : { "weight_quant": weight_quant },
            "encoder.encoders.1.attn_layers.2.attention.value_projection.weight": { "weight_quant": weight_quant },
            "encoder.encoders.1.attn_layers.2.attention.out_projection.weight"  : { "weight_quant": weight_quant },
            "encoder.encoders.1.attn_layers.2.conv1.weight"                    : { "weight_quant": weight_quant },
            "encoder.encoders.1.attn_layers.2.conv2.weight"                    : { "weight_quant": weight_quant },
            
            # Decoder
            # Decoder 0 attn 0
            "decoder.layers.0.self_attention.query_projection.weight": { "weight_quant": weight_quant },
            "decoder.layers.0.self_attention.key_projection.weight"  : { "weight_quant": weight_quant },
            "decoder.layers.0.self_attention.value_projection.weight": { "weight_quant": weight_quant },
            "decoder.layers.0.self_attention.out_projection.weight"  : { "weight_quant": weight_quant },
            "decoder.layers.0.cross_attention.query_projection.weight": { "weight_quant": weight_quant },
            "decoder.layers.0.cross_attention.key_projection.weight"  : { "weight_quant": weight_quant },
            "decoder.layers.0.cross_attention.value_projection.weight": { "weight_quant": weight_quant },
            "decoder.layers.0.cross_attention.out_projection.weight"  : { "weight_quant": weight_quant },
            "decoder.layers.0.conv1.weight"                          : { "weight_quant": weight_quant },
            "decoder.layers.0.conv2.weight"                          : { "weight_quant": weight_quant },
            
            # Decoder 1 attn 0
            "decoder.layers.1.self_attention.query_projection.weight": { "weight_quant": weight_quant },
            "decoder.layers.1.self_attention.key_projection.weight"  : { "weight_quant": weight_quant },
            "decoder.layers.1.self_attention.value_projection.weight": { "weight_quant": weight_quant },
            "decoder.layers.1.self_attention.out_projection.weight"  : { "weight_quant": weight_quant },
            "decoder.layers.1.cross_attention.query_projection.weight": { "weight_quant": weight_quant },
            "decoder.layers.1.cross_attention.key_projection.weight"  : { "weight_quant": weight_quant },
            "decoder.layers.1.cross_attention.value_projection.weight": { "weight_quant": weight_quant },
            "decoder.layers.1.cross_attention.out_projection.weight"  : { "weight_quant": weight_quant },
            "decoder.layers.1.conv1.weight"                          : { "weight_quant": weight_quant },
            "decoder.layers.1.conv2.weight"                          : { "weight_quant": weight_quant },
            
            # Decoder 2 attn 0
            "decoder.layers.2.self_attention.query_projection.weight": { "weight_quant": weight_quant },
            "decoder.layers.2.self_attention.key_projection.weight"  : { "weight_quant": weight_quant },
            "decoder.layers.2.self_attention.value_projection.weight": { "weight_quant": weight_quant },
            "decoder.layers.2.self_attention.out_projection.weight"  : { "weight_quant": weight_quant },
            "decoder.layers.2.cross_attention.query_projection.weight": { "weight_quant": weight_quant },
            "decoder.layers.2.cross_attention.key_projection.weight"  : { "weight_quant": weight_quant },
            "decoder.layers.2.cross_attention.value_projection.weight": { "weight_quant": weight_quant },   
            "decoder.layers.2.cross_attention.out_projection.weight"  : { "weight_quant": weight_quant },
            "decoder.layers.2.conv1.weight"                          : { "weight_quant": weight_quant },
            "decoder.layers.2.conv2.weight"                          : { "weight_quant": weight_quant },
            }
            # fmt: on
            optimizer = torch.optim.Adam(model.parameters(), lr=1, eps=1e-9)

            optimizer = OptimLP(
                optimizer,
                quantizationSettings,
                model.named_parameters(),
            )

            if torch.cuda.is_available():
                state = torch.load(model_filename_pretrained, map_location=device)
            else:
                state = torch.load(model_filename_pretrained, map_location="cpu")
            model.load_state_dict(state["model_state_dict"])

            optimizer.step()

            loss = run_validation(model, get_dataset(config), device)
            data_np = [[tensor.cpu().numpy()] for tensor in loss]
            loss_list.append([data_np, wl])

            print(f"Quantizing with wl: {wl}, fl: {fl}", flush=True)
            print(f"Loss: {loss}", flush=True)

            # Save loss_list as pickle
        with open(f"loss_list_{rounding}.pkl", "wb") as f:
            pickle.dump(loss_list, f)
