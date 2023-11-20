
import numpy as np
from pycsou.operator import SquaredL2Norm
from deconv import PsfStencil
from utils import downsample_volume, epfl_deconv_data
from deconv import trim_psf
import pycsou.runtime as pycrt
from pycsou.opt.solver import PD3O
from pycsou.opt.stop import MaxIter, RelError
from pycsou.operator import PositiveL1Norm, L21Norm
from pycsou.operator import Convolve, Gradient

import dask.array as da
from pathlib import Path

gpu = False  # Lazy loading on GPU is not implemented yet
λ1 = 1e-1
λ2 = 1e-4

fname = "big_deconv.zarr"
fdir = "data/epfl_data/"
y_path = Path(fdir, "y_" + fname)
psf_path = Path(fdir, "psf_" + fname)
x_path = Path(fdir, "x_" + fname)

if not (y_path.exists() and psf_path.exists()):
    y, psf = [], []
    for channel in range(3):
        y_, psf_ = epfl_deconv_data(channel)
        y_ = downsample_volume(y_, 1)
        psf_ = downsample_volume(psf_, 1)

        # Same preprocessing as in Scico
        y_ -= y_.min()
        y_ /= y_.max()
        psf_ /= psf_.sum()

        y.append(y_)
        psf.append(psf_)

    y = np.stack(y)
    psf = np.stack(psf)

    chunks = (1, 89, 84, 26)
    y = da.from_array(y, chunks=chunks)
    y.to_zarr(y_path, overwrite=True)
    psf = da.from_array(psf, chunks=chunks)
    psf.to_zarr(psf_path, overwrite=True)

y = da.from_zarr(y_path)
psf = da.from_zarr(psf_path)
psf = psf.compute()


# Stopping criterion
default_stop_crit = (
    RelError(eps=1e-3, var="x", f=None, norm=2, satisfy_all=True)
    & RelError(eps=1e-3, var="z", f=None, norm=2, satisfy_all=True)
    & MaxIter(20)
) | MaxIter(500)

with pycrt.Precision(pycrt.Width.SINGLE):

    forwards = [PsfStencil(psf_, percentile=99.99, ) for psf_ in psf]

    l21_norm = L21Norm(arg_shape=(3, *y.shape), l2_axis=(0, 1))

    grad = Gradient(arg_shape=y.shape[1:],
                    diff_method="fd",
                    accuracy=4,
                    sampling=[0.64, 0.64, 1.6],
                    dtype="float32",
                    gpu=gpu)

    grad.lipschitz(tight=False)
    posl1 = PositiveL1Norm(dim=y[0].size)

    x_recons_tv = []

    for y_channel, forward in zip(y, forwards):
        sl2 = SquaredL2Norm(dim=y_channel.size).asloss(y_channel.ravel())
        sl2.diff_lipschitz()

        l21_norm = L21Norm(arg_shape=(3, *y_channel.shape), l2_axis=(0,))

        loss = sl2 * forward
        loss.diff_lipschitz()

        solver = PD3O(
            f=loss, g=λ1 * posl1, h= λ2 * l21_norm, K=grad, show_progress=True, verbosity=20
        )
        # Fit
        solver.fit(x0=0 * y_channel.ravel(), tuning_strategy=2, stop_crit=default_stop_crit)
        x_recons_tv.append(solver.solution().reshape(y_channel.shape))

x_recons_tv = da.stack(x_recons_tv)
x_recons_tv.to_zarr(x_path, overwrite=True)
print("Done!")