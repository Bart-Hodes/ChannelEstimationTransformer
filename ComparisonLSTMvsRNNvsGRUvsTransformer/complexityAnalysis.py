import torch

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Informer.model import InformerStack
from models.Transformer.model import Transformer, build_transformer
from models.RNN.model import RNN
from models.LSTM.model import LSTM
from models.GRU.model import GRU
from config import get_config
from ptflops import get_model_complexity_info

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_model(model_type, config):
    model_map = {
        "GRU": GRU,
        "LSTM": LSTM,
        "RNN": RNN,
        "Transformer": build_transformer,
        "Informer": InformerStack,
    }
    if model_type in model_map:
        model_cls = model_map[model_type]
        if model_type == "Informer":
            model = model_cls(
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
        elif model_type == "Transformer":
            model = model_cls(
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
        else:
            model = model_cls(
                config["enc_in"], config["enc_in"], config["hs"], config["hl"]
            )
        return model.to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def input_constructor(input_shape, config, model_type):
    batch_size, seq_len, N = input_shape
    encoder_input = torch.zeros((batch_size, seq_len, N)).to(device)
    if model_type in ["Transformer", "Informer"]:
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

        if model_type == "Informer":
            inputs = {
                "x_enc": encoder_input,
                "x_dec": decoder_input,
                "x_mark_enc": torch.zeros_like(encoder_input).to(device),
                "x_mark_dec": torch.zeros_like(decoder_input).to(device),
            }

        else:
            inputs = {
                "encoder_input": encoder_input,
                "decoder_input": decoder_input,
            }
    else:
        inputs = torch.randn(batch_size, seq_len, N).to(device)
    return inputs


def get_model_stats(model_type, config):
    model = get_model(model_type, config)
    total_params = sum(p.numel() for p in model.parameters())

    input_shape = (
        (1, config["seq_len"], 16) if model_type != "GRU" else (1, 30, config["enc_in"])
    )

    if model_type == "Informer" or model_type == "Transformer":
        backend = "aten"
    else:
        backend = "pytorch"
    macs, params = get_model_complexity_info(
        model,
        input_shape,
        as_strings=True,
        backend=backend,
        input_constructor=lambda x: input_constructor(x, config, model_type),
    )
    return model_type, total_params, macs, params


if __name__ == "__main__":
    config = get_config()
    models_to_evaluate = ["Transformer", "Informer", "GRU", "LSTM", "RNN"]

    print("Model Statistics:")
    stats_list = []
    for model_type in models_to_evaluate:
        model_stats = get_model_stats(model_type, config)
        stats_list.append(model_stats)

    for idx, (model_type, total_params, macs, params) in enumerate(stats_list, start=1):
        print(f"\n{idx}. {model_type}")
        print(f"Number of parameters {model_type}: {total_params}")
        print(f"Computational complexity: {macs}")
        print(f"Number of parameters: {params}")
