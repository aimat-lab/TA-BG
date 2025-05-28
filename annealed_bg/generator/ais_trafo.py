import bgflow as bg
import numpy as np
import torch

from annealed_bg.systems.base import System


def to_AIS_space(
    x: torch.Tensor,
    left: float = 0.0,
    right: float = 1.0,
    scale: float = 10.0,
    unbound_space: bool = True,
):
    if unbound_space:
        # Scale from [left, right] to [-1,1]:
        x = (x - left) / (right - left)
        x = (x - 0.5) * 2

        # Apply the inverse of the tanh function => transform to unbound space:
        # First, get the jacobian of this trafo: d/dx tanh(x) = 1-tanh^2(x)
        # => d/dx atanh(x) = 1/(1-x^2)
        log_jac_det = torch.sum(torch.log(1 / (1 - x**2)), dim=1)
        x = torch.atanh(x)

        x = (
            x / 2 + 0.5
        )  # Scale back to [0,1] (at least for the linear part of the atanh)
        x = x * (right - left) + left
    else:
        log_jac_det = 0.0

    x = x * scale
    log_jac_det = log_jac_det + np.log(scale) * x.shape[1]

    return x, log_jac_det


def from_AIS_space(
    x: torch.Tensor,
    left: float = 0.0,
    right: float = 1.0,
    scale: float = 10.0,
    unbound_space: bool = True,
):
    x = x / scale
    log_jac_det = -np.log(scale) * x.shape[1]

    if unbound_space:
        x = (x - left) / (right - left)
        x = (x - 0.5) * 2
        x = torch.tanh(x)
        log_jac_det = log_jac_det + torch.sum(torch.log(1 - x**2), dim=1)
        x = x / 2 + 0.5
        x = x * (right - left) + left

    return x, log_jac_det


class AISTrafo(bg.Flow):
    def __init__(
        self,
        system: System,
        AIS_trafo_config: dict[str, tuple[bool, float]] | None = None,
    ):
        super().__init__()

        self._AIS_trafo_config = AIS_trafo_config
        self._IC_shape_info = system.IC_shape_info

        self._spline_range = system.spline_range

        assert (
            self._spline_range is not None
        ), "Spline range must be set for unbound trafo."

    def forward(self, *xs, inverse=False, **kwargs) -> tuple:
        output_xs = []
        log_jac_det = torch.zeros(xs[0].shape[0], 1, device=xs[0].device)

        if not inverse:
            transform_fn = to_AIS_space
        else:
            transform_fn = from_AIS_space

        if self._AIS_trafo_config is not None:
            for i, channel_name in enumerate(self._IC_shape_info.names()):
                if channel_name in self._AIS_trafo_config:
                    channel_config = self._AIS_trafo_config[channel_name]
                    transformed_x, new_log_jac_det = transform_fn(
                        xs[i],
                        left=self._spline_range[0],
                        right=self._spline_range[1],
                        unbound_space=channel_config[0],
                        scale=channel_config[1],
                    )
                    output_xs.append(transformed_x)
                    log_jac_det += (
                        new_log_jac_det.view(-1, 1)
                        if isinstance(new_log_jac_det, torch.Tensor)
                        else new_log_jac_det
                    )
                else:
                    output_xs.append(xs[i])
            return *output_xs, log_jac_det
        else:
            return *xs, log_jac_det

    def _forward(self, *xs, **kwargs):
        return self.forward(*xs, inverse=False, **kwargs)

    def _inverse(self, *xs, **kwargs):
        return self.forward(*xs, inverse=True, **kwargs)
