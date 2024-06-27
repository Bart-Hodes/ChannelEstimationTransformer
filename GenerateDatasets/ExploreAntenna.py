import numpy as np
import pickle
import torch
import scipy.io as scio
from sionna.mimo import StreamManagement
import os

from sionna.ofdm import ResourceGrid

from sionna.channel.tr38901 import AntennaArray, CDL
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel

import matplotlib.pyplot as plt

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


carrier_frequency = 28e9  # Carrier frequency in Hz.
# This is needed here to define the antenna element spacing.

ut_array = AntennaArray(
    num_rows=1,
    num_cols=4,
    polarization="single",
    polarization_type="cross",
    antenna_pattern="38.901",
    carrier_frequency=carrier_frequency,
)

ut_array.show()
plt.savefig("ut_array.png")
ut_array.show_element_radiation_pattern()
plt.savefig("ut_array_element_radiation_pattern.png")

bs_array = AntennaArray(
    num_rows=1,
    num_cols=int(num_bs_ant / 2),
    polarization="dual",
    polarization_type="cross",
    antenna_pattern="38.901",
    carrier_frequency=carrier_frequency,
)


delay_spread = 100e-9  # Nominal delay spread in [s]. Please see the CDL documentation
# about how to choose this value.

direction = "downlink"  # The `direction` determines if the UT or BS is transmitting.
# In the `uplink`, the UT is transmitting.
cdl_model = "B"  # Suitable values are ["A", "B", "C", "D", "E"]

speed = 30  # UT speed [km/u]. BSs are always assumed to be fixed.
# The direction of travel will chosen randomly within the x-y plane.


# Configure a channel impulse reponse (CIR) generator for the CDL model.
# cdl() will generate CIRs that can be converted to discrete time or discrete frequency.
cdl = CDL(
    cdl_model,
    delay_spread,
    carrier_frequency,
    ut_array,
    bs_array,
    direction,
    min_speed=speed / 3.6,
    max_speed=speed / 3.6,
)

number_of_slots = 100
batch_size = 256
num_of_batches = 100
