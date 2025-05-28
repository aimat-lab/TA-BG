from abc import ABC, abstractmethod
from typing import Callable, Literal

import bgflow as bg
import torch


class System(ABC):
    def __init__(
        self,
        energy_fn: Callable,
        event_shapes: torch.Size,
        energy_bridge=None,
    ):
        self._energy_fn = energy_fn
        self.event_shapes = event_shapes
        self._bridge = energy_bridge

    def energy(self, xs: torch.Tensor, temperature: float | None = None):
        return self._energy_fn(xs, temperature)

    @property
    @abstractmethod
    def IC_shape_info(self) -> bg.ShapeDictionary:
        pass

    @property
    @abstractmethod
    def spline_range(self) -> tuple[float, float] | None:
        pass

    @abstractmethod
    def get_prior_type_and_kwargs(
        self, transform_type: Literal["spline", "rnvp"]
    ) -> tuple[dict, dict]:
        pass

    @abstractmethod
    def plot_IC_marginals(
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
        pass
