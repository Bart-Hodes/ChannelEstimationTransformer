import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from torch.nn.common_types import _size_1_t
from typing import Union, Tuple, Optional, List
from torch.nn.modules.utils import _single, _reverse_repeat_tuple

import time

import numpy as np

from .fibbinary_cpu.python_cpp_wrapper import closest_fibbinary, closest_fibbinary_array


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x, fibbinary):

    # print(fibbinary)
    # Convert fibbinary to a Tensor so that in the cpp module I can use the torch API
    fibbinary_tensor = torch.tensor(fibbinary, dtype=torch.int32)

    # print(x)
    # print(fibbinary_tensor)
    # Apperently python is very slow so we use the option to use cpp/cuda modules
    # From 7 sec per batch to 0.1 sec per batch
    closest_values = closest_fibbinary_array(x, fibbinary_tensor)

    # print(x)
    # print(closest_values)

    y = closest_values
    # Create a gradient-compatible tensor
    y_grad = x

    # Adjust the result to retain gradients
    result = y - y_grad.detach() + y_grad

    return result


def fibbinary():
    x = 0
    while True:
        yield x
        y = ~(x >> 1)
        x = (x - y) & y


class LinearLSQ(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LinearLSQ, self).__init__()

        self.nbits = None

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.register_parameter("step_size", None)
        self.quantize = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #     nn.init.uniform_(self.bias, -bound, bound)

        if self.nbits is not None:
            self.Qn = -(2 ** (self.nbits - 1))
            self.Qp = 2 ** (self.nbits - 1) - 1
            self.step_size = nn.Parameter(self.weight.abs().mean() / math.sqrt(self.Qp))
            self.g = 1.0 / math.sqrt(self.weight.numel() * self.Qp)

            self.fibgen = fibbinary()
            self.fibcodebook = []
            fib = next(self.fibgen)

            while fib.bit_length() <= self.nbits - 1:
                self.fibcodebook.append(fib)
                fib = next(self.fibgen)
            result = []
            for num in self.fibcodebook:
                result.append(num)
                if num != 0:
                    result.append(-num)
            self.fibcodebook = result

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.quantize:
            step_size = grad_scale(self.step_size, self.g)
            # print(self.weight)
            # print(step_size)
            # print(self.Qn)
            # print(self.Qp)
            # print(self.weight / step_size)
            # print((self.weight / step_size).clamp(self.Qn, self.Qp))
            # print(round_pass(
            #         (self.weight / step_size).clamp(self.Qn, self.Qp), self.fibcodebook
            #     ))
            w_q = (
                round_pass(
                    (self.weight / step_size).clamp(self.Qn, self.Qp), self.fibcodebook
                )
                * step_size
            )
            return F.linear(input, w_q, self.bias)
        else:
            return F.linear(input, self.weight, self.bias)


class _ConvNd(nn.Module):

    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    def _conv_forward(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:  # type: ignore[empty-body]
        ...

    in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: torch.Tensor
    bias: Optional[torch.Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
        dilation: Tuple[int, ...],
        transposed: bool,
        output_padding: Tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}"
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )

        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.nbits = None
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
                self.padding, 2
            )

        if transposed:
            self.weight = nn.Parameter(
                torch.empty(
                    (in_channels, out_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    (out_channels, in_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.register_parameter("step_size", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        #     if fan_in != 0:
        #         bound = 1 / math.sqrt(fan_in)
        #         nn.init.uniform_(self.bias, -bound, bound)

        if self.nbits is not None:
            self.Qn = -(2 ** (self.nbits - 1))
            self.Qp = 2 ** (self.nbits - 1) - 1
            self.step_size = nn.Parameter(self.weight.abs().mean() / math.sqrt(self.Qp))
            self.g = 1.0 / math.sqrt(self.weight.numel() * self.Qp)

            self.fibgen = fibbinary()
            self.fibcodebook = []
            fib = next(self.fibgen)
            while fib.bit_length() <= self.nbits - 1:
                self.fibcodebook.append(fib)
                fib = next(self.fibgen)
            result = []
            for num in self.fibcodebook:
                result.append(num)
                if num != 0:
                    result.append(-num)
            self.fibcodebook = result

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"


class Conv1dLSQ(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # we create new variables below to make mypy happy since kernel_size has
        # type Union[int, Tuple[int]] and kernel_size_ has type Tuple[int]
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        self.nbits = None
        self.quantize = False
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _single(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _conv_forward(
        self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ):
        if self.padding_mode != "zeros":
            return F.conv1d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _single(0),
                self.dilation,
                self.groups,
            )
        return F.conv1d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.quantize:
            step_size = grad_scale(self.step_size, self.g)
            w_q = (
                round_pass(
                    (self.weight / step_size).clamp(self.Qn, self.Qp), self.fibcodebook
                )
                * step_size
            )
            return self._conv_forward(input, w_q, self.bias)
        else:
            return self._conv_forward(input, self.weight, self.bias)
