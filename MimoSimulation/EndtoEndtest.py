import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
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
from sionna.utils.metrics import compute_ber


# Define the number of UT and BS antennas.
# For the CDL model, that will be used in this notebook, only
# a single UT and BS are supported.
num_ut = 1
num_bs = 1
num_ut_ant = 2
num_bs_ant = 4
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


rg = ResourceGrid(
    num_ofdm_symbols=14,
    fft_size=76,
    subcarrier_spacing=15e3,
    num_tx=1,
    num_streams_per_tx=num_streams_per_tx,
    cyclic_prefix_length=6,
    num_guard_carriers=[5, 6],
    dc_null=True,
    pilot_pattern="kronecker",
    pilot_ofdm_symbol_indices=[0],
)


carrier_frequency = 28e9  # Carrier frequency in Hz.
# This is needed here to define the antenna element spacing.

delay_spread = 30e-9  # Nominal delay spread in [s]. Please see the CDL documentation
# about how to choose this value.

direction = "uplink"  # The `direction` determines if the UT or BS is transmitting.
# In the `uplink`, the UT is transmitting.
cdl_model = "B"  # Suitable values are ["A", "B", "C", "D", "E"]

speed = 30  # UT speed [m/s]. BSs are always assumed to be fixed.
# The direction of travel will chosen randomly within the x-y plane.


ut_array = AntennaArray(
    num_rows=1,
    num_cols=int(num_ut_ant / 2),
    polarization="dual",
    polarization_type="cross",
    antenna_pattern="38.901",
    carrier_frequency=carrier_frequency,
)

bs_array = AntennaArray(
    num_rows=1,
    num_cols=int(num_bs_ant / 2),
    polarization="dual",
    polarization_type="cross",
    antenna_pattern="38.901",
    carrier_frequency=carrier_frequency,
)


# Configure a channel impulse reponse (CIR) generator for the CDL model.
# cdl() will generate CIRs that can be converted to discrete time or discrete frequency.
cdl = CDL(
    cdl_model,
    delay_spread,
    carrier_frequency,
    ut_array,
    bs_array,
    direction,
    min_speed=speed,
)

l_min, l_max = time_lag_discrete_time_channel(rg.bandwidth)
l_tot = l_max - l_min + 1

frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)


num_bits_per_symbol = 2  # QPSK modulation
coderate = 0.5  # Code rate
n = int(rg.num_data_symbols * num_bits_per_symbol)  # Number of coded bits
k = int(n * coderate)  # Number of information bits

# The binary source will create batches of information bits
binary_source = BinarySource()

# The encoder maps information bits to coded bits
encoder = LDPC5GEncoder(k, n)

# The mapper maps blocks of information bits to constellation symbols
mapper = Mapper("qam", num_bits_per_symbol)

# The resource grid mapper maps symbols onto an OFDM resource grid
rg_mapper = ResourceGridMapper(rg)

# The zero forcing precoder precodes the transmit stream towards the intended antennas
zf_precoder = ZFPrecoder(rg, sm, return_effective_channel=True)

# OFDM modulator and demodulator
modulator = OFDMModulator(rg.cyclic_prefix_length)
demodulator = OFDMDemodulator(rg.fft_size, l_min, rg.cyclic_prefix_length)

# This function removes nulled subcarriers from any tensor having the shape of a resource grid
remove_nulled_scs = RemoveNulledSubcarriers(rg)

# The LS channel estimator will provide channel estimates and error variances
ls_est = LSChannelEstimator(rg, interpolation_type="nn")

# The LMMSE equalizer will provide soft symbols together with noise variance estimates
lmmse_equ = LMMSEEqualizer(rg, sm)

# The demapper produces LLR for all coded bits
demapper = Demapper("app", "qam", num_bits_per_symbol)

# The decoder provides hard-decisions on the information bits
decoder = LDPC5GDecoder(encoder, hard_out=True)


batch_size = 1  # We pick a small batch_size as executing this code in Eager mode could consume a lot of memory
ebno_db = 8
perfect_csi = True

no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)
b = binary_source([batch_size, 1, rg.num_streams_per_tx, encoder.k])
c = encoder(b)
x = mapper(c)
x_rg = rg_mapper(x)

# The following values for truncation are recommended.
# Please feel free to tailor them to you needs.

a, tau = cdl(
    batch_size=batch_size,
    num_time_steps=rg.num_time_samples + l_tot - 1,
    sampling_frequency=rg.bandwidth,
)

# OFDM modulation with cyclic prefix insertion
x_time = modulator(x_rg)

# Compute the discrete-time channel impulse reponse
h_time = cir_to_time_channel(rg.bandwidth, a, tau, l_min, l_max, normalize=True)

apply_channel = ApplyTimeChannel(rg.num_time_samples, l_tot, add_awgn=True)

y_time = apply_channel([x_time, h_time, no])

# OFDM demodulation and cyclic prefix removal
y = demodulator(y_time)

perfect_csi = True
if perfect_csi:

    # We need to sub-sample the channel impulse reponse to compute perfect CSI
    # for the receiver as it only needs one channel realization per OFDM symbol
    a_freq = a[
        ..., rg.cyclic_prefix_length : -1 : (rg.fft_size + rg.cyclic_prefix_length)
    ]
    a_freq = a_freq[..., : rg.num_ofdm_symbols]

    # Compute the channel frequency response
    h_freq = cir_to_ofdm_channel(frequencies, a_freq, tau, normalize=True)

    h_hat, err_var = remove_nulled_scs(h_freq), 0.0
else:
    h_hat, err_var = ls_est([y, no])

print(h_hat)
plt.plot(np.real(h_hat)[0, 0, 0, 0, 0, :, 0])

plt.show()
print(h_hat)
x_hat, no_eff = lmmse_equ([y, h_hat, err_var, no])
llr = demapper([x_hat, no_eff])
b_hat = decoder(llr)
ber = compute_ber(b, b_hat)

print("BER: {}".format(ber))
