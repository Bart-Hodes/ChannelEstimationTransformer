import torch
from torch.utils.data import Dataset

import scipy.io as sio

from json import JSONEncoder
import json

from InformerModel.model import InformerStack

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


class EncodeTensor(JSONEncoder, Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(EncodeTensor, self).default(obj)


state = torch.load(
    "weights/wl2_fl1_epoch_49.pt",
    map_location=torch.device("cpu"),
)
model.load_state_dict(state["model_state_dict"])

for key in model.state_dict():
    with open(f"weight_export/{key}.json", "w") as json_file:
        json.dump(model.state_dict()[key], json_file, cls=EncodeTensor)

# for key in model.state_dict():
#     key_sanitized = key.replace(".", "_")
#     mat_contents = {key_sanitized: model.state_dict()[key].cpu().detach().numpy()}
#     sio.savemat(f"weight_export/{key_sanitized}.mat", mat_contents)

# with open("torch_weights.json", "w") as json_file:
#     json.dump(model.state_dict(), json_file, cls=EncodeTensor)
