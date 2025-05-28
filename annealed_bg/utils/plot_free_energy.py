from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from pyemma.plots.plots2d import _to_free_energy


def plot_1D_marginal(
    xs: np.ndarray,
    weights: np.ndarray | None = None,
    plot_as_free_energy: bool = False,
    ax: plt.Axes | None = None,
    n_bins: int = 100,
    label: str | None = None,
    linestyle: str = "-",
    return_data: bool = False,
):
    hist, edges = np.histogram(xs, bins=n_bins, density=True, weights=weights)

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    if plot_as_free_energy:
        nonzero = hist > 0
        free_energy = np.inf * np.ones(shape=hist.shape)
        free_energy[nonzero] = -np.log(hist[nonzero])
        free_energy[nonzero] -= np.min(free_energy[nonzero])

        ax.plot(
            (edges[:-1] + edges[1:]) / 2, free_energy, label=label, linestyle=linestyle
        )
    else:
        ax.plot((edges[:-1] + edges[1:]) / 2, hist, label=label, linestyle=linestyle)

    if return_data:
        return (edges[:-1] + edges[1:]) / 2, (
            hist if not plot_as_free_energy else free_energy
        )


def _get_histogram(
    xall: np.ndarray,
    yall: np.ndarray,
    nbins: int | Sequence[int],
    weights: np.ndarray = None,
    range: np.ndarray | None = None,
):
    """Compute a two-dimensional histogram.

    Args:
        xall: Sample x-coordinates.
        yall: Sample y-coordinates.
        nbins: Number of histogram bins used in each dimension.
        weights: Sample weights. If None, all samples have the same weight.
        range: The leftmost and rightmost edges of the bins along each dimension [[xmin, xmax], [ymin, ymax]].

    Returns:
        x: The bins' x-coordinates in meshgrid format.
        y: The bins' y-coordinates in meshgrid format.
        z: Histogram counts in meshgrid format.
    """

    z, xedge, yedge = np.histogram2d(
        xall, yall, bins=nbins, weights=weights, range=range
    )
    x = 0.5 * (xedge[:-1] + xedge[1:])
    y = 0.5 * (yedge[:-1] + yedge[1:])

    return x, y, z.T  # transpose to match x/y-directions


def plot_2D_free_energy(
    xall: np.ndarray,
    yall: np.ndarray,
    weights: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    nbins: int | Sequence[int] = 100,
    minener_zero: bool = True,
    kT: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "nipy_spectral",
    cbar: bool = True,
    cbar_label: str = r"free energy / $kT$",
    cax: plt.Axes | None = None,
    cbar_orientation: str = "vertical",
    print_max_f: bool = False,
    range: np.ndarray | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a two-dimensional free energy map using a histogram of
    scattered data.

    Args:
        xall: Sample x-coordinates.
        yall: Sample y-coordinates.
        weights: Sample weights. If None, all samples have the same weight.
        ax: The ax to plot to; if ax=None, a new ax (and fig) is created.
        nbins: Number of histogram bins used in each dimension.
        minener_zero: Shifts the energy minimum to zero.
        kT: The value of kT in the desired energy unit. By default, energies are
            computed in kT (setting 1.0). If you want to measure the energy in
            kJ/mol at 298 K, use kT=2.479 and change the cbar_label accordingly.
        vmin: Lowest free energy value to be plotted.
        vmax: Highest free energy value to be plotted.
        cmap: The color map to use.
        cbar: Plot a color bar.
        cbar_label: Colorbar label string; use None to suppress it.
        cax: Plot the colorbar into a custom axes object instead of stealing space
            from ax.
        cbar_orientation: Colorbar orientation; choose 'vertical' or 'horizontal'.
        print_max_f: Print the maximum free energy value to the console.
        range: The range of the histogram. If None, the range is computed from the data.

    Returns:
        fig: The figure in which the used ax resides.
        ax: The ax in which the map was plotted.
    """

    if minener_zero and vmin is None:
        vmin = 0.0

    x, y, z = _get_histogram(
        xall,
        yall,
        nbins=nbins,
        weights=weights,
        range=range,
    )
    f = _to_free_energy(z, minener_zero=minener_zero) * kT

    if print_max_f:
        print("Max free energy: ", np.max(f[~np.isinf(f)]))

    if vmax is not None:
        f[f >= vmax] = np.inf

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = ax.get_figure()

    map = ax.imshow(
        f,
        extent=[
            x.min() if range is None else range[0][0],
            x.max() if range is None else range[0][1],
            y.min() if range is None else range[1][0],
            y.max() if range is None else range[1][1],
        ],
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    if cbar:
        if cax is None:
            cbar_ = fig.colorbar(map, ax=ax, orientation=cbar_orientation)
        else:
            cbar_ = fig.colorbar(map, cax=cax, orientation=cbar_orientation)
        if cbar_label is not None:
            cbar_.set_label(cbar_label)

    return fig, ax
