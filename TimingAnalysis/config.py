from pathlib import Path


def get_config():
    return {
        "batch_size": 1,
        "num_epochs": 10,
        "SNR": 80,
        "lr": 1e-3,
        "seq_len": 90,
        "pred_len": 5,
        "label_len": 10,  # Updated label length
        "d_model": 128,
        "enc_in": 16,  # Added enc_in
        "dec_in": 16,  # Added dec_in
        "c_out": 16,  # Added c_out
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
        "factor": 5,  # I moved this line here assuming it belongs to this config
        "hs": 256,
        "hl": 4,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "99",
        "dataset_name": "Seq_Len_100_Beamforming_CDLB",
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
