from pathlib import Path


def get_config():
    return {
        "batch_size": 256,
        "num_epochs": 100,
        "SNR": 21,
        "lr": 1e-3,
        "seq_len": 90,
        "pred_len": 5,
        "label_len": 10,
        "d_model": 128,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "dataset_name": "Seq_Len_100_Beamforming_CDLB",
        "experiment_name": "runs/tmodelBATCHNORMTEST",
        "num_bits": 8,
        "wl": None,
        "fl": None,
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
