from typing import Literal

import bgflow as bg
import matplotlib.pyplot as plt
import numpy as np
import torch
from fab.target_distributions.gmm import GMM
from fab.utils.plotting import plot_contours, plot_marginal_pair

from annealed_bg.systems.base import System
from annealed_bg.utils.wandb import log_figure


def create_FAB_gmm_target() -> GMM:
    torch.manual_seed(0)  # To get the same system as in FAB
    target = GMM(dim=2, n_mixes=40, loc_scaling=40, log_var_scaling=1.0)
    return target


class GMMSystem(System):
    def __init__(
        self,
        system_temp: float,
    ):
        target = create_FAB_gmm_target()
        target.to("cuda")

        def wrapped_energy_fn(xs, temperature=None):
            if temperature is None:
                return -1.0 * target.log_prob(xs).view(-1, 1) * 1.0 / system_temp
            else:
                return -1.0 * target.log_prob(xs).view(-1, 1) * 1.0 / temperature

        super().__init__(wrapped_energy_fn, torch.Size([2]), energy_bridge=None)

        self._IC_shape_info = bg.ShapeDictionary()
        self._IC_shape_info[bg.TARGET] = (2,)

    @property
    def IC_shape_info(self) -> bg.ShapeDictionary:
        return self._IC_shape_info

    @property
    def spline_range(self) -> tuple[float, float] | None:
        return [-50.0, 50.0]

    def get_prior_type_and_kwargs(
        self, transform_type: Literal["spline", "rnvp"]
    ) -> tuple[dict, dict]:

        prior_type = dict()
        prior_kwargs = dict()

        if transform_type == "rnvp":
            prior_type[bg.TARGET] = bg.NormalDistribution
        elif transform_type == "spline":
            prior_type[bg.TARGET] = bg.TruncatedNormalDistribution
            prior_kwargs[bg.TARGET] = {
                "mu": 0.0,
                "sigma": 10.0,
                "lower_bound": -50.0,
                "upper_bound": 50.0,
            }

        return prior_type, prior_kwargs

    def plot_IC_marginals(
        self,
        flow_data: torch.Tensor | None,
        ground_truth_data: torch.Tensor,
        tag: str,
        current_i: int | None = None,
        flow_data_weights: torch.Tensor | None = None,
        data_range: list | None = None,
        plot_as_free_energy: bool = False,
        marginals_2D: list | None = None,
        marginals_2D_vmax: float = 6.5,
        dpi_marginals=50,
        dpi_2D_marginal=100,
        report_wandb: bool = False,
        output_dir: str | None = None,
        do_calculate_2D_marginal_kld: bool = False,
        is_big_eval: bool = False,
    ):
        if flow_data_weights is not None:
            # Resample the dataset for visualization of the reweighted distribution.

            # Switch to numpy to avoid 2^24 elements limit of torch.multinomial:
            weights_np = np.float64(flow_data_weights.view(-1).numpy())

            weights_np /= np.sum(weights_np)

            indices = np.random.choice(
                np.arange(flow_data_weights.size(0)),
                size=flow_data_weights.size(0),
                replace=True,
                p=weights_np,
            )
            indices = torch.from_numpy(indices)

            flow_data = flow_data[indices]

        alpha = 0.3
        loc_scaling = 40

        plotting_bounds = (-loc_scaling * 1.4, loc_scaling * 1.4)

        plt.set_cmap("viridis")

        fig, ax = plt.subplots()

        plot_contours(
            lambda x: -1.0 * self.energy(x.cuda()).detach().cpu(),
            bounds=plotting_bounds,
            ax=ax,
            n_contour_levels=50,
            grid_width_n_points=200,
        )
        plot_marginal_pair(flow_data, ax=ax, bounds=plotting_bounds, alpha=alpha)

        log_figure(
            tag=f"{tag}_scatter",
            current_i=current_i,
            fig=plt.gcf(),
            write_pickle=True,
            report_wandb=report_wandb,
            output_dir=output_dir,
            dpi=300,
        )
