import gc
import traceback
from time import time
from typing import Tuple

import torch
import wandb
from bgflow import BoltzmannGenerator
from bgflow.nn.flow.sequential import SequentialFlow

from annealed_bg.config.system import SystemConfig
from annealed_bg.evaluation.metrics import calculate_reverse_ESS
from annealed_bg.evaluation.plot import plot_tica
from annealed_bg.evaluation.utils import process_tica_in_batches
from annealed_bg.systems.base import System
from annealed_bg.systems.molecular.system import MolecularSystem


def evaluate_sampling(
    generator: BoltzmannGenerator,
    IC_trafo: SequentialFlow,
    n_samples: int,
    val_datasets_IC: dict,
    val_datasets_cart: dict,
    eval_sampling_T_pairs: list[Tuple[float | None, float | None]],
    main_temp: float,
    current_i: int,
    system: System,
    system_cfg: SystemConfig,
    report_wandb: bool = True,
    output_dir: str | None = None,
    do_calculate_2D_marginal_kld: bool = True,
    do_calculate_ESS: bool = True,
    is_big_eval: bool = False,
):
    """Evaluate the sampling of a generator: Generate plots (1D marginals + 2D marginals + TICA) and calculate reverse ESS + 2Dmarginal KLD.

    Args:
        generator: The generator to evaluate.
        IC_trafo: IC trafo.
        n_samples: The number of samples to use for evaluation per temperature.
        val_datasets_IC: The validation datasets (IC space).
        val_datasets_cart: The validation datasets (cartesian space).
        eval_sampling_T_pairs: The temperatures to evaluate at. Same structure as in config.py:
            List of (temperature to sample at (None means boundary temperature), temperature to reweight to (None means no reweighting)).
        main_temp: The boundary temperature.
        current_i: The current iteration in the training loop.
        system: The system object.
        system_cfg: The system config object.
        report_wandb: Whether to report to W&B. If False, we print the metrics to the stdout.
        output_dir: The output directory. If None, no output is saved to disk.
        do_calculate_2D_marginal_kld: Whether to calculate the 2D marginal KLD.
        do_calculate_ESS: Whether to calculate the reverse ESS.
        is_big_eval: Whether this is a big evaluation. If true, we also plot 1D marginals for the molecular systems.
    """

    reweighting_Ts_per_sampling_T = {}  # {T_sample: [ ... ]}
    for sampling_T, reweighting_T in eval_sampling_T_pairs:
        if sampling_T not in reweighting_Ts_per_sampling_T:
            reweighting_Ts_per_sampling_T[sampling_T] = []
        reweighting_Ts_per_sampling_T[sampling_T].append(reweighting_T)

    for sampling_T in reweighting_Ts_per_sampling_T:

        start = time()
        print(
            "Starting sampling evaluation at T =",
            sampling_T,
            "using",
            n_samples,
            "samples",
            "reweighting to temperatures",
            reweighting_Ts_per_sampling_T[sampling_T],
        )

        eval_batch_size = int(2**12)

        flow_samples_ICs = []
        for i in range(len(val_datasets_IC[main_temp])):
            flow_samples_ICs.append(
                torch.empty((n_samples, val_datasets_IC[main_temp][i].shape[1]))
            )
        flow_samples_cart = torch.empty(
            (n_samples, val_datasets_cart[main_temp].shape[1])
        )

        log_weights_per_reweighting_T = {
            T: torch.empty((n_samples, 1))
            for T in reweighting_Ts_per_sampling_T[sampling_T]
        }

        with torch.no_grad():
            for i in range(0, n_samples, eval_batch_size):
                n_samples_batch = min(eval_batch_size, n_samples - i)

                samples = None
                log_qs = None
                for _ in range(10):
                    try:
                        if sampling_T is None:
                            context = None
                        else:
                            context = (
                                torch.ones(n_samples_batch, 1, device="cuda")
                                * sampling_T
                            )

                        samples, bg_energies = generator.sample(
                            n_samples_batch, with_energy=True, context=context
                        )
                        log_qs = -bg_energies

                        break
                    except Exception as e:
                        print(
                            f"Exception caught in sample generation while evaluating flow:"
                        )
                        # Print the full stacktrace:
                        print(traceback.format_exc())
                        print("Trying again...")

                if samples is None:
                    print(
                        "Failed to generate samples 10 times. Cancelling sampling evaluation."
                    )
                    return

                samples_IC = IC_trafo.forward(samples, inverse=True)
                samples_IC = [s.cpu() for s in samples_IC[:-1]]  # exclude log_det

                samples_cpu = samples.cpu()

                flow_samples_cart[i : i + n_samples_batch] = samples_cpu
                for j in range(len(flow_samples_ICs)):
                    flow_samples_ICs[j][i : i + n_samples_batch] = samples_IC[j]

                del samples_cpu, samples_IC

                for reweighting_T in reweighting_Ts_per_sampling_T[sampling_T]:
                    if reweighting_T is None:
                        log_weights_per_reweighting_T[reweighting_T][
                            i : i + n_samples_batch
                        ] = torch.zeros(n_samples_batch, 1)
                    else:
                        energies_reweighting_T = generator._target.energy(
                            samples, temperature=reweighting_T
                        )

                        log_weights_per_reweighting_T[reweighting_T][
                            i : i + n_samples_batch
                        ] = (-log_qs - energies_reweighting_T).cpu()

                del samples  # Free GPU memory
                # Run garbage collection explicitly every 500K samples
                if i % 500000 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        ##### Generate all plots for this sampling temperature #####

        for reweighting_T in reweighting_Ts_per_sampling_T[sampling_T]:

            log_weights = log_weights_per_reweighting_T[reweighting_T]
            assert log_weights.shape[0] == n_samples
            assert log_weights.shape[1] == 1
            log_weights = log_weights.view(-1)

            if do_calculate_ESS and reweighting_T is not None:
                ESS = calculate_reverse_ESS(
                    log_weights=log_weights,
                    clipping_fraction=system_cfg.eval_IS_clipping_fraction,
                )

                if report_wandb:
                    wandb.log(
                        {
                            (
                                f"ESS_"
                                + (
                                    f"{float(sampling_T)}K"
                                    if sampling_T is not None
                                    else "None"
                                )
                                + f"_{float(reweighting_T)}K"
                            ): ESS,
                        },
                        step=current_i,
                    )
                else:
                    print(
                        f"ESS_"
                        + (
                            f"{float(sampling_T)}K"
                            if sampling_T is not None
                            else "None"
                        )
                        + f"_{float(reweighting_T)}K"
                        + ":",
                        ESS,
                    )

            if system_cfg.eval_IS_clipping_fraction is not None:
                # Clipping of IS weights, see Flow AIS Bootstrap paper:
                k = int(log_weights.shape[0] * system_cfg.eval_IS_clipping_fraction)
                if k < 2:
                    print(
                        "Warning: k < 2, no weight clipping for 2D marginals is performed."
                    )
                else:
                    clip_value = torch.min(torch.topk(log_weights, k).values)
                    log_weights[log_weights > clip_value] = clip_value

            log_weights = log_weights - torch.max(log_weights)
            weights = torch.exp(log_weights)

            # Get the ground truth data:
            ground_truth_T = reweighting_T if reweighting_T is not None else sampling_T
            if sampling_T is None and reweighting_T is None:
                ground_truth_T = main_temp

            ground_truth_ICs = val_datasets_IC[ground_truth_T]
            downsampled_ground_truth_ICs = []
            for i in range(len(ground_truth_ICs)):
                downsampled_ground_truth_ICs.append(ground_truth_ICs[i][:n_samples])

            print(
                "Using",
                downsampled_ground_truth_ICs[0].shape[0],
                "of",
                val_datasets_IC[ground_truth_T][0].shape[0],
                "ground truth samples for evaluation at T =",
                ground_truth_T,
            )

            reweighting_temps_tag = (
                f"{float(sampling_T)}K" if sampling_T is not None else "None"
            ) + (f"_{float(reweighting_T)}K" if reweighting_T is not None else "")

            ##### Plot 1D marginals and 2D marginals #####
            for plot_as_free_energy in (
                [False, True] if isinstance(system, MolecularSystem) else [False]
            ):
                for flow_data, ground_truth_data, tag, current_marginals_2D in zip(
                    flow_samples_ICs,
                    downsampled_ground_truth_ICs,
                    [
                        reweighting_temps_tag + "_" + channel_name
                        for channel_name in system.IC_shape_info.names()
                    ],
                    [
                        (
                            system_cfg.marginals_2D[channel_name]
                            if channel_name in system_cfg.marginals_2D
                            else None
                        )
                        for channel_name in system.IC_shape_info.names()
                    ],
                ):
                    system.plot_IC_marginals(
                        flow_data=flow_data,
                        ground_truth_data=ground_truth_data,
                        tag=tag if not plot_as_free_energy else f"{tag}_F",
                        current_i=current_i,
                        flow_data_weights=(
                            weights if reweighting_T is not None else None
                        ),
                        plot_as_free_energy=plot_as_free_energy,
                        marginals_2D=(
                            current_marginals_2D if plot_as_free_energy else None
                        ),  # Only plot once
                        marginals_2D_vmax=system_cfg.marginals_2D_vmax,
                        report_wandb=report_wandb,
                        output_dir=output_dir,
                        do_calculate_2D_marginal_kld=do_calculate_2D_marginal_kld,
                        is_big_eval=is_big_eval,
                    )

            ##### Plot tica #####
            if system_cfg.tica_path_and_selection is not None and isinstance(
                system, MolecularSystem
            ):
                ground_truth_cart = val_datasets_cart[ground_truth_T][:n_samples]
                ticas_ground_truth = process_tica_in_batches(
                    ground_truth_cart,
                    system.openmm_system.mdtraj_topology,
                    system_cfg.tica_path_and_selection[0],
                    selection=system_cfg.tica_path_and_selection[1],
                    eigs_kept=2,
                )
                ticas_flow = process_tica_in_batches(
                    flow_samples_cart,
                    system.openmm_system.mdtraj_topology,
                    system_cfg.tica_path_and_selection[0],
                    selection=system_cfg.tica_path_and_selection[1],
                    eigs_kept=2,
                )

                plot_tica(
                    ticas_flow=ticas_flow,
                    ticas_ground_truth=ticas_ground_truth,
                    flow_data_weights=weights,
                    tag=(f"{float(sampling_T)}K" if sampling_T is not None else "None")
                    + (
                        f"_{float(reweighting_T)}K" if reweighting_T is not None else ""
                    ),
                    current_i=current_i,
                    dpi=100,
                    report_wandb=report_wandb,
                    output_dir=output_dir,
                    vmax=system_cfg.tica_vmax,
                )

            del log_weights, weights

        del (flow_samples_cart,)

        for i in reversed(range(len(flow_samples_ICs))):
            del flow_samples_ICs[i]

        print(
            "Evaluation at T =",
            sampling_T,
            "took",
            time() - start,
            "s",
        )
