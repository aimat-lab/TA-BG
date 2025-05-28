import numpy as np
import torch
import torch.nn.functional as F


def calculate_2D_marginal_kld(
    flow_data: torch.Tensor,
    ground_truth_data: torch.Tensor,
    flow_data_weights: torch.Tensor | None = None,
) -> float:
    """Calculate the forward KLD between the 2D marginal of the ground truth and the flow data."""

    assert len(flow_data.shape) == 2 and flow_data.shape[1] == 2
    assert len(ground_truth_data.shape) == 2 and ground_truth_data.shape[1] == 2
    if flow_data_weights is not None:
        assert (
            len(flow_data_weights.shape) == 1
            and flow_data_weights.shape[0] == flow_data.shape[0]
        )
        flow_data_weights = flow_data_weights / torch.sum(flow_data_weights)

    # We try to follow Midgley et al. (2023) as closely as possible:

    nbins_ram = 100
    eps_ram = 1e-10

    hist_ram_ground_truth = np.histogram2d(
        ground_truth_data[:, 0].numpy() * 2 * np.pi,
        ground_truth_data[:, 1].numpy() * 2 * np.pi,
        nbins_ram,
        range=[[0, 2 * np.pi], [0, 2 * np.pi]],
        density=True,
    )[0]
    hist_ram_flow = np.histogram2d(
        flow_data[:, 0].numpy() * 2 * np.pi,
        flow_data[:, 1].numpy() * 2 * np.pi,
        nbins_ram,
        range=[[0, 2 * np.pi], [0, 2 * np.pi]],
        density=True,
        weights=(flow_data_weights.numpy() if flow_data_weights is not None else None),
    )[0]

    kld_ram = (
        np.sum(
            hist_ram_ground_truth
            * np.log((hist_ram_ground_truth + eps_ram) / (hist_ram_flow + eps_ram))
        )
        * (2 * np.pi / nbins_ram) ** 2  # To get the properly normalized integral / KLD
    )

    return kld_ram


def calculate_reverse_ESS(
    log_weights: torch.Tensor, clipping_fraction: float | None = 1.0e-4
) -> float:
    log_weights_copy = log_weights.clone()
    assert len(log_weights_copy.shape) == 1

    if clipping_fraction is not None:
        k = int(log_weights_copy.shape[0] * clipping_fraction)
        if k > 1:
            clip_value = torch.min(torch.topk(log_weights_copy, k).values)
            log_weights_copy[log_weights_copy > clip_value] = clip_value

    normalized_weights = F.softmax(log_weights_copy, dim=0)
    ESS = 1 / torch.sum(normalized_weights**2) / normalized_weights.shape[0]

    return ESS.item()


# Source: https://github.com/lollcat/se3-augmented-coupling-flows
def calculate_forward_ESS(log_w: torch.Tensor) -> torch.Tensor:
    log_z_inv = torch.logsumexp(-log_w, dim=0) - np.log(log_w.shape[0])
    log_z_expectation_p_over_q = torch.logsumexp(log_w, dim=0) - np.log(log_w.shape[0])
    log_forward_ess = -log_z_inv - log_z_expectation_p_over_q
    forward_ess = torch.exp(log_forward_ess)
    return forward_ess
