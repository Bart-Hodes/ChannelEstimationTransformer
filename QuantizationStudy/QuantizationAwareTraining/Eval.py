import torch
from torch.utils.data import DataLoader

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


from qtorch.optim import OptimLP

from qtorch.quant import fixed_point_quantize_partial, fixed_point_quantize

print("test", flush=True)

import numpy as np

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

    return loss / len(val_dataloader)


if __name__ == "__main__":

    print("Starting training", flush=True)
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
        weight_quant_setting = args.weigth_quant_setting
    if args.rounding is not None:
        rounding = args.rounding
        if args.weigth_quant_setting == "partial":
            if rounding == "proximal":
                config["steps"] = [0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.98,0.99, 0.995,0.998,0.999,0.9995,0.9998,0.9999,1.0]
                # config["steps"] = [0.0556,0.1111,0.1667,0.2222,0.2778,0.3333,0.3889,0.4444,0.5000,0.5556,0.6111,0.6667,0.7222,0.7778,0.8333,0.8889,0.9444,1.0000]
            elif rounding == "stochastic":
                config["steps"] = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5, 0.55,0.60,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
            elif rounding == "distant":
                config["steps"] = [0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.15, 0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
            config["num_epochs"] = 10
        else:
            config["steps"] = [1.0]
            config["num_epochs"] = 70

    # fmt:on
    config["model_filename"] = "Weights/tmodel_pretrained.pt"
    config["experiment_name"] = f"runs/{weight_quant_setting}/{rounding}"

    loss_list = []
    for wl in [2]:
        fl = wl - 1
        config["wl"] = wl
        config["fl"] = fl
        print(f"Quantizing with wl: {wl}, fl: {fl}")

        if weight_quant_setting == "partial":
            weight_quant = [
                lambda x, percentage: fixed_point_quantize_partial(
                    x, wl=wl, fl=fl, percentage=percentage, rounding=rounding
                ),
                "partial",
            ]
        else:
            weight_quant = [
                lambda x, percentage: fixed_point_quantize(
                    x, wl=wl, fl=fl, rounding=rounding
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
        print(f"Quantizing with {rounding} rounding mode")
        print(f"Quantizing with wl: {wl}, fl: {fl}")

        # fmt: on

        model = get_model(config)
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
        loss = run_validation(quantizationSettings, config)
        data_np = [[tensor.cpu().numpy()] for tensor in loss]

        loss_list.append([data_np, wl])

        # Save the loss list to a file
        import pickle

        # Save the loss list to a file using pickle
        with open(
            f"losstestsetsetsets_list_{weight_quant_setting}_{rounding}.pickle", "wb"
        ) as f:
            pickle.dump(loss_list, f)
