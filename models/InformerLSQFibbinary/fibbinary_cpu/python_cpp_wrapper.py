import torch
from torch.utils.cpp_extension import load
import os

current_path = os.path.dirname(os.path.realpath(__file__))
# Load the custom C++ extension
fibbinary_cpu = load(name='fibbinary_cpu', sources=[os.path.join(current_path, 'cpp_fibbinary_functions.cpp')])

if torch.cuda.is_available():
    fibbinary_gpu = load(name='fibbinary_gpu', sources=[os.path.join(current_path, 'cpp_fibbinary_functions.cu')])
else:
    fibbinary_gpu = fibbinary_cpu

def get_module(x):
    if x.is_cuda:
        fibbinary_module = fibbinary_gpu
    else:
        fibbinary_module = fibbinary_cpu
    return fibbinary_module

# Define a Python function to call the C++ function
def closest_fibbinary(val):
    fibbinary_ext = get_module(val)
    return fibbinary_ext.closest_fibbinary(val)



def closest_fibbinary_array(val, fibbinary):
    fibbinary_ext = get_module(val)
    if(val.dim() == 2):
        return fibbinary_ext.closest_fibbinary_array_2d(val, fibbinary)
    if(val.dim() == 3):
        return fibbinary_ext.closest_fibbinary_array_3d(val, fibbinary)
    else:
        exception = "Input tensor must be 2D or 3D, but got " + str(val.dim()) + "D"
