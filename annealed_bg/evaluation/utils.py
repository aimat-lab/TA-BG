from typing import Callable, List

import mdtraj
import torch
from bgflow import BoltzmannGenerator

from annealed_bg.evaluation.metrics import calculate_forward_ESS
from annealed_bg.utils.tica import to_tics


def evaluate_metric_batched(
    metric_fn: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
    data: torch.Tensor,
    context_temp: float | None = None,
) -> float:
    batch_size_eval = int(2**13)
    n_samples = data.shape[0]
    metric_sum = 0.0

    if context_temp is None:
        context = None
    else:
        context = torch.ones(batch_size_eval, 1, device="cuda") * context_temp

    for i in range(0, n_samples, batch_size_eval):
        n_samples_batch = min(batch_size_eval, n_samples - i)
        batch = data[i : i + n_samples_batch]
        batch = batch.cuda()

        with torch.no_grad():
            metric_sum += (
                metric_fn(
                    batch,
                    context=(
                        context[:n_samples_batch, ...] if context is not None else None
                    ),
                )
                .mean()
                .cpu()
                .item()
                * n_samples_batch
            )

    metric_sum /= n_samples
    return metric_sum


def evaluate_forward_ESS_batched(
    generator: BoltzmannGenerator,
    temp: float,
    val_data_cart: torch.Tensor,
    clip_fractions: List[float | None] = [None, 1e-3, 1e-4],
) -> dict[float | None, float]:
    eval_batch_size = int(2**12)
    context = torch.ones(eval_batch_size, 1, device="cuda") * temp

    log_ws = torch.empty((val_data_cart.shape[0],))

    for i in range(0, val_data_cart.shape[0], eval_batch_size):
        n_samples_batch = min(eval_batch_size, val_data_cart.shape[0] - i)
        batch = val_data_cart[i : i + n_samples_batch]
        batch = batch.cuda()

        with torch.no_grad():
            log_q = -generator.energy(batch, context=context[:n_samples_batch, ...])
            log_p = -generator._target.energy(batch, temperature=temp)

            log_ws[i : i + n_samples_batch] = (log_p - log_q).cpu().squeeze()

    all_ESS = {}
    for clip_fraction in clip_fractions:

        current_log_ws = log_ws.clone()
        if clip_fraction is not None:
            k = int(clip_fraction * val_data_cart.shape[0])
            clip_value = torch.min(torch.topk(current_log_ws, k).values)
            current_log_ws[current_log_ws > clip_value] = clip_value

        forward_ESS = calculate_forward_ESS(current_log_ws).item()

        all_ESS[clip_fraction] = forward_ESS

    return all_ESS


def process_tica_in_batches(
    data: torch.Tensor,
    topology: mdtraj.Topology,
    tica_path: str,
    selection: str,
    eigs_kept: int = 2,
    batch_size: int = 50000,
) -> torch.Tensor:
    tica_results = []
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        tica_batch = to_tics(
            batch, topology, tica_path, selection=selection, eigs_kept=eigs_kept
        )
        tica_results.append(tica_batch)
        del batch, tica_batch

    return torch.cat(tica_results, dim=0)
