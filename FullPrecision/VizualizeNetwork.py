import torch
from torch.utils.data import DataLoader
from pathlib import Path
import warnings
import datetime
from InformerModel.model import InformerStack
from dataset import SeqData, LoadBatch
from config import get_config
import torchviz

current_datetime = datetime.datetime.now()
print("Current Date and Time:", current_datetime)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_dataset(config):
    evaluateDatasetName = (
        f'../GenerateDatasets/Datasets/{config["dataset_name"]}__validate.pickle'
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

    return evaluaterLoader


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


def count_weights(model):
    print("Weights in each module:")
    total_params = 0
    for name, module in model.named_modules():
        num_params = sum(p.numel() for p in module.parameters())
        total_params += num_params
        print(f"{name}: {num_params} parameters")
    print(f"Total number of parameters: {total_params}")


def export_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(config)
    val_dataloader = get_dataset(config)

    model.eval()

    # Count the weights
    count_weights(model)

    # Use a batch from the evaluation dataset as input data
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
        break

    # Ensure the input is on the correct device
    data = data.to(device)

    # Export the model
    encoder_input_range = torch.arange(config["seq_len"]).to(device)
    decoder_input_range = torch.arange(config["pred_len"] + config["label_len"]).to(
        device
    )

    model_output = model(
        encoder_input,
        encoder_input_range,
        decoder_input,
        decoder_input_range,
    )

    if isinstance(model_output, (tuple, list)):
        model_output = model_output[0]

    # Use torchviz to visualize the model with modules and parameters
    dot = torchviz.make_dot(
        model_output, params={name: param for name, param in model.named_parameters()}
    )
    dot.format = "png"
    dot.render("model_visualization")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()

    model_name = f"ei_{config['enc_in']}_di_{config['dec_in']}_co_{config['c_out']}_sl_{config['seq_len']}_ll_{config['label_len']}_pl_{config['pred_len']}_f_{config['factor']}_dm_{config['d_model']}_nh_{config['n_heads']}_el_{config['e_layers']}_dl_{config['d_layers']}_df_{config['d_ff']}_do_{config['dropout']}_at_{config['attn']}_em_{config['embed']}_ac_{config['activation']}"
    print("Model_name: ", model_name)

    export_model(config)
