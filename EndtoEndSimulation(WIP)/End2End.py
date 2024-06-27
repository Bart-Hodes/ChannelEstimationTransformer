import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy

import tensorflow as tf

from sionna.mimo import StreamManagement

from sionna.ofdm import (
    ResourceGrid,
    ResourceGridMapper,
    LSChannelEstimator,
    LMMSEEqualizer,
)
from sionna.ofdm import (
    OFDMModulator,
    OFDMDemodulator,
    ZFPrecoder,
    RemoveNulledSubcarriers,
)

from sionna.channel.tr38901 import AntennaArray, CDL, Antenna
from sionna.channel import (
    subcarrier_frequencies,
    cir_to_ofdm_channel,
    cir_to_time_channel,
    time_lag_discrete_time_channel,
)
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel, TimeChannel

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

from sionna.mapping import Mapper, Demapper

from sionna.utils import BinarySource, ebnodb2no, sim_ber
from sionna.utils.metrics import compute_ber, compute_bler

from torch.utils.data import Dataset, DataLoader

from metrics import NMSELoss
from dataset import SeqData, LoadBatch
from config import get_config

import warnings

from models.model import InformerStack

# Define the number of UT and BS antennas.
num_ut = 1
num_bs = 1
num_ut_ant = 2
num_bs_ant = 64
# The number of transmitted streams is equal to the number of UT antennas
# in both uplink and downlink
num_streams_per_tx = num_ut_ant

# Create an RX-TX association matrix
# rx_tx_association[i,j]=1 means that receiver i gets at least one stream
# from transmitter j. Depending on the transmission direction (uplink or downlink),
# the role of UT and BS can change. However, as we have only a single
# transmitter and receiver, this does not matter:
rx_tx_association = np.array([[1]])

# Instantiate a StreamManagement object
# This determines which data streams are determined for which receiver.
# In this simple setup, this is fairly easy. However, it can get more involved
# for simulations with many transmitters and receivers.
sm = StreamManagement(rx_tx_association, num_streams_per_tx)

rg = ResourceGrid(  # The `num_ofdm_symbols` parameter is used to specify the number of OFDM symbols in
    # the resource grid. OFDM (Orthogonal Frequency Division Multiplexing) is a
    # modulation technique commonly used in wireless communication systems. Each OFDM
    # symbol consists of multiple subcarriers, and the number of subcarriers is
    # determined by the FFT size. The `num_ofdm_symbols` parameter determines how many
    # OFDM symbols are included in the resource grid, which affects the overall duration
    # of the transmitted signal.
    num_ofdm_symbols=75,
    fft_size=1,
    subcarrier_spacing=120e3,
    num_tx=1,
    num_streams_per_tx=num_streams_per_tx,
    cyclic_prefix_length=0,
    num_guard_carriers=[0, 0],
    dc_null=True,
    pilot_pattern="kronecker",
    pilot_ofdm_symbol_indices=[0],
)


number_of_slots = 100
batch_size = 64
num_of_batches = 100

# Encoding parameters
num_bits_per_symbol = 2
coderate = 0.5

# Channel Estimation
# channel_estimation = "test"
channel_estimation = "Transformer"

# Initialize parameters and helper functions
n = int(rg.num_data_symbols * num_bits_per_symbol)
k = int(n * coderate)

criterion = NMSELoss()

binary_source = BinarySource()
encoder = LDPC5GEncoder(k, n)
mapper = Mapper("qam", num_bits_per_symbol)
rg_mapper = ResourceGridMapper(rg)
zf_precoder = ZFPrecoder(rg, sm, return_effective_channel=True)
channel_freq = ApplyOFDMChannel(add_awgn=True)
ls_est = LSChannelEstimator(rg, interpolation_type="nn")
lmmse_equ = LMMSEEqualizer(rg, sm)
demapper = Demapper("app", "qam", num_bits_per_symbol)
decoder = LDPC5GDecoder(encoder, hard_out=True)
remove_nulled_scs = RemoveNulledSubcarriers(rg)


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


def predictChannel(config):
    transformer = get_model(config)

    transformer_dict_name = f"TrainedTransformers/Freq_{transformer.__class__.__name__}_{channel_model}_{factor}x_d_model{d_model}_n_heads{n_heads}_e_layers{e_layers}_d_layers{d_layers}_d_ff{d_ff}_dropout{dropout}_attn_{attn}_embed_{embed}_activation_{activation}_enc_in{enc_in}_dec_in{dec_in}_c_out{c_out}_seq_len{seq_len}_label_len{label_len}_pred_len{pred_len}_output_attention{output_attention}_distil{distil}_pilotInterval_{pilotInterval}_{direction}.pt"
    state_dict = torch.load(transformer_dict_name, map_location=torch.device("cpu"))
    state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    transformer.load_state_dict(state_dict)
    transformer = torch.nn.DataParallel(transformer).cuda() if use_gpu else transformer
    print("transformer has been loaded!")


SNRS = [*range(15, 30, 3)]
BER = np.zeros(len(SNRS))
BLER = np.zeros(len(SNRS))

print(SNRS)

for k, ebno_db in enumerate(SNRS):
    evaluateDatasetName = (
        f'../GenerateDatasets/Datasets/{config["dataset_name"]}__validate.pickle'
    )

    evaluateData = SeqData(
        evaluateDatasetName, config["seq_len"], config["pred_len"], config["SNR"]
    )
    evaluaterLoader = DataLoader(
        dataset=evaluateData,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    avg_ber = 0
    avg_bler = 0

    for batch_idx, batch in enumerate(evaluaterLoader):
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)

        H, H_noise, H_seq, H_pred = batch

        x_rg, g = zf_precoder([x_rg, h_est_slot])
        y = channel_freq([x_rg, h_freq_slot, no])

        h_hat, err_var = ls_est([y, no])

        x_hat, no_eff = lmmse_equ([y, h_hat, err_var, no])
        llr = demapper([x_hat, no_eff])
        b_hat = decoder(llr)

        symbol_ber = compute_ber(b, b_hat)
        symbol_bler = compute_bler(b, b_hat)

        ber += symbol_ber / (number_of_slots - pred_len - seq_len + 1)
        bler += symbol_bler / (number_of_slots - pred_len - seq_len + 1)

        avg_ber += ber
        avg_bler += bler

    print(f"SNR: {ebno_db}")
    print(f"The Total Bit Error Rate (BER) is: {avg_ber/num_iter:.8f}")
    print(f"The Total Block Error Rate (BER) is: {avg_bler/num_iter:.8f}")
    BER[k] = avg_ber
    BLER[k] = avg_bler


# plt.figure()
# plt.plot(SNRS,BER)
# plt.plot(SNRS,BLER)
# plt.title(f"Coderate {coderate}")
# plt.legend(["BER","BLER"])
# plt.savefig(f"Plots/BER_BLER_{pilotInterval}_{direction}", dpi=300)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()

    # if(channel_estimation == "Transformer"):
    #     h_est_fft = []
    #     for i in range(number_of_slots - pred_len - seq_len + 1):

    #         h_freq_prediction_indexed = h_freq_prediction[:,i: seq_len+ i,...]
    #         data = LoadBatch(torch.from_numpy(h_freq_prediction_indexed))
    #         inp_net = data.to(device)

    #         enc_inp = inp_net
    #         dec_inp =  torch.zeros_like( enc_inp[:, -5:, :] ).to(device)
    #         dec_inp =  torch.cat([enc_inp[:, seq_len - label_len:seq_len, :], dec_inp], dim=1)

    #         outputs_informer = transformer(enc_inp, dec_inp)[0]
    #         outputs_informer = outputs_informer[:,:pred_len,:]
    #         outputs_informer = outputs_informer.cpu().detach()

    #         # Reshape into sionna format
    #         outputs_informer = real2complex(np.array(outputs_informer))[:,pred_len-1,:]
    #         if direction == "uplink":
    #             outputs_informer = outputs_informer.reshape([batch_size,4,2])
    #         else:
    #             outputs_informer = outputs_informer.reshape([batch_size,2,4])

    #         # outputs_informer= tf.transpose(outputs_informer, perm=[0,2,1])

    #         if(i == 0):
    #             h_est_fft = np.expand_dims(outputs_informer, axis=1)
    #         else:
    #             h_est_fft = np.append(h_est_fft,np.expand_dims(outputs_informer, axis=1),axis=1)

    # if(fft_index == 0):
    #     h_est = np.expand_dims(h_est_fft, axis=4)
    # else:
    #     h_est = np.append(h_est, np.expand_dims(h_est_fft, axis=4), axis=4)
