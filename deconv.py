import time as t
import warnings
from functools import partial

# +
import numpy as np
import scipy as sp
import scipy.signal
import skimage.io
from pycsou.abc import DiffFunc, DiffMap, LinOp, Map, ProxFunc
from pycsou.operator import SquaredL2Norm
from pycsou.operator.interop import from_sciop, from_source, from_torch
from pycsou.operator.interop.torch import *
from pycsou.runtime import Precision, Width, enforce_precision
from pycsou.util import get_array_module, to_NUMPY
from pycsou.operator import DiagonalOp, FFT , Pad, SubSample, Sum, Convolve
from pycsou.util import view_as_complex, view_as_real
from numpy.fft import fftshift, fft, ifft
from pycsou.operator import block_diag, hstack, vstack
import pycsou.util as pycu
import pycsou.abc as pyca

from utils import downsample_volume, epfl_deconv_data
import pycsou.runtime as pycrt
from pycsou.opt.solver import PD3O
from pycsou.opt.stop import MaxIter, RelError
from pycsou.operator import Jacobian, PositiveL1Norm, L21Norm
from pycsou.operator import Gradient, Sum

class Roll(pyca.UnitOp):
    def __init__(self, arg_shape, axes=None, shift=None):
        self.arg_shape = arg_shape
        if axes is None:
            self.axes = tuple(range(len(arg_shape)))
        if shift is None:
            self.shift = [dim // 2 for dim in arg_shape]

        self.shift_adjoint = [-sh for sh in self.shift]
        dim = np.prod(arg_shape).item()
        super().__init__(shape=(dim, dim))

    def apply(self, arr):
        sh = arr.shape[:-1]
        arr = arr.reshape(*sh, *self.arg_shape)
        xp = pycu.get_array_module(arr)
        return xp.roll(arr, self.shift, self.axes).reshape(*sh, -1)

    def adjoint(self, arr):
        sh = arr.shape[:-1]
        arr = arr.reshape(*sh, *self.arg_shape)
        xp = pycu.get_array_module(arr)
        return xp.roll(arr, self.shift_adjoint, self.axes).reshape(*sh, -1)


def ComplexMult(arr):
    mask_r = SubSample((arr.size,), slice(0, None, 2))
    mask_i = SubSample((arr.size,), slice(1, None, 2))

    arr_r = DiagonalOp(mask_r(arr))
    arr_i = DiagonalOp(mask_i(arr))

    # First compute real part
    real = mask_r.T * (arr_r * mask_r - arr_i * mask_i)
    # Second compute imaginary part
    imag = mask_i.T * (arr_i * mask_r + arr_r * mask_i)

    return real + imag


def PsfFourier(psf):
    # PSF and input arrays are assumed to have both the same shape
    shape = np.array(psf.shape)
    # Both are padded to have 2 N - 1 shape
    size = shape * 2 - 1
    # fsize = 2 ** np.ceil(np.log2(size)).astype(int)
    pad = Pad(arg_shape=psf.shape, pad_width=[(sh, sh) for sh in (size - shape) // 2])
    ft = FFT(arg_shape=pad._pad_shape, real=True)
    fft_f = ft * pad
    psf_fft = fft_f(psf.ravel())
    mult = ComplexMult(psf_fft)

    fft_shape = np.array(shape) * 2
    roll = Roll(pad._pad_shape)

    startind = (pad._pad_shape - shape) // 2
    endind = startind + shape
    slices = [slice(startind[k], endind[k]) for k in range(len(endind))]
    center = SubSample(pad._pad_shape, *slices)

    op = (1 / ft.dim) * center * roll * ft.T * mult * fft_f

    setattr(op, "center", center)
    setattr(op, "roll", roll)
    setattr(op, "ft", ft)
    setattr(op, "mult", mult)
    setattr(op, "fft_f", fft_f)
    setattr(op, "pad", pad)

    return op


def PsfStencil(psf, percentile=None):
    # PSF and input arrays are assumed to have both the same shape
    # If percentile_kept is 99, only 1 % of the PSF is kept

    arg_shape = psf.shape

    # Trim PSF
    if percentile is not None:
        psf, center = trim_psf(psf, percentile)
        print(f"PSF trimmed from shape {arg_shape} (size: {np.prod(arg_shape).item()}) to {psf.shape} ({psf.size}).")

    op = Convolve(
        arg_shape=arg_shape,
        kernel=psf,
        center=center,
        mode="constant",
        enable_warnings=True,
    )
    return op


def PsfFourierChunks(psf, chunk_shape, percentile=None):
    # PSF and input arrays are assumed to have both the same shape

    # If percentile_kept is 99, only 1 % of the PSF is kept
    # Trim PSF
    arg_shape = psf.shape
    if percentile is not None:
        psf = trim_psf(psf, percentile, centered=True)
        print(f"PSF trimmed from shape {arg_shape} (size: {np.prod(arg_shape).item()}) to {psf.shape} ({psf.size}).")

    shape = np.array(psf.shape)
    # Both are psf and arr padded
    size = shape + np.array(chunk_shape) - 1
    # fsize = 2 ** np.ceil(np.log2(size)).astype(int)
    pad_psf = Pad(arg_shape=psf.shape, pad_width=[(sh // 2, sh // 2 + sh % 2) for sh in (size - shape) ])
    pad_arr = Pad(arg_shape=chunk_shape, pad_width=[(sh // 2, sh // 2 + sh % 2) for sh in (size - chunk_shape)])
    ft = FFT(arg_shape=size, real=True)

    psf_fft = ft(pad_psf(psf.ravel()))
    mult = ComplexMult(psf_fft)

    fft_shape = np.array(shape) * 2
    roll = Roll(size)

    startind = (size - shape) // 2
    endind = startind + shape
    slices = [slice(startind[k], endind[k]) for k in range(len(endind))]
    recenter = SubSample(size, *slices)

    op = (1 / ft.dim) * recenter * roll * ft.T * mult * ft * pad_arr
    return op

def trim_psf(psf, percentile, centered=False):
    # For stencils, it should ideally be odd (we need an origin)
    # For fourier, it should ideally be the same as the original
    slices = []
    center = []
    psf_ = psf.copy()
    psf_[psf_ < np.percentile(psf_.ravel(), percentile)] = 0.
    for j in range(3):
        ids = [0, 1, 2]
        ids.remove(j)
        psf_sum = psf_.sum(tuple(ids))
        trim_f = len(np.trim_zeros(psf_sum, trim='f'))
        trim_b = len(np.trim_zeros(psf_sum, trim='b'))
        if centered:
            trim_f = trim_b = (trim_f + trim_b) // 2
        slices.append(slice(len(psf_sum) - trim_f, trim_b))
        center.append(len(psf_sum)//2 - (len(psf_sum) - trim_f))
    psf = psf[tuple(slices)]
    psf /= psf.sum()
    if centered:
        return psf
    else:
        return psf, center
