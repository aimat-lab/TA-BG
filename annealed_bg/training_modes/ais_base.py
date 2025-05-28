import os
from time import time
from typing import Dict, List, Tuple

import bgflow as bg
import torch
import wandb
from bgflow import BoltzmannGenerator
from bgflow.nn.flow.sequential import SequentialFlow
from fab.sampling_methods import AnnealedImportanceSampler
from fab.sampling_methods.transition_operators import HamiltonianMonteCarlo

from annealed_bg.config.evaluation import EvaluationConfig
from annealed_bg.config.system import SystemConfig
from annealed_bg.config.training_modes.ais_base import AISBaseTrainingModeConfig
from annealed_bg.evaluation.metrics import calculate_reverse_ESS
from annealed_bg.generator.ais_trafo import AISTrafo
from annealed_bg.systems.base import System
from annealed_bg.systems.molecular.system import MolecularSystem
from annealed_bg.training_modes.base import TrainingMode
from annealed_bg.training_modes.fab_util import BaseDistributionWrapper

##### Parts of this code are adapted from https://github.com/lollcat/fab-torch #####


class AISBaseTrainingMode(TrainingMode):
    def __init__(
        self,
        config: AISBaseTrainingModeConfig,
        generator_IC: BoltzmannGenerator,
        IC_trafo: SequentialFlow,
        system: System,
        main_temp: float,
    ):
        self._energy_calls_counter = 0

        self._ais_base_config = config
        self.generator_IC = generator_IC
        self.IC_trafo = IC_trafo
        self.system = system
        self._main_temp = main_temp

        self.dims = [item[0] for item in system.IC_shape_info.values()]
        self.dim = sum([item[0] for item in system.IC_shape_info.values()])

        ##### Set up AIS trafo that is applied to the ICs before performing AIS #####

        if isinstance(system, MolecularSystem):
            NO_constraint_layers = system.NO_constraint_layers
        else:
            NO_constraint_layers = None

        AIS_trafo = AISTrafo(
            system=system,
            AIS_trafo_config=config.AIS_trafo_config,
        )
        self.inv_AIS_trafo = bg.InverseFlow(AIS_trafo)

        self.generator_IC = BoltzmannGenerator(
            prior=self.generator_IC._prior,
            flow=SequentialFlow(
                (
                    self.generator_IC._flow._blocks[:-NO_constraint_layers]
                    if NO_constraint_layers is not None
                    else self.generator_IC._flow._blocks
                )
                + torch.nn.ModuleList([AIS_trafo]),
                context_preprocessor=self.generator_IC._flow._context_preprocessor,
            ),
            target=self.generator_IC._target,
        )
        self.IC_trafo = SequentialFlow(
            torch.nn.ModuleList([self.inv_AIS_trafo])
            + (
                generator_IC._flow._blocks[-NO_constraint_layers:]
                if NO_constraint_layers is not None
                else []
            )
            + self.IC_trafo._blocks,
        )

        self.constrain_trafo = SequentialFlow(
            generator_IC._flow._blocks[-NO_constraint_layers:]
            if NO_constraint_layers is not None
            else []
        )

        ##### Set up the transition operator + AIS sampler #####

        if config.reject_spline_OOB_samples:
            sampling_bounds = torch.empty((self.dim, 2))
            sampling_bounds[:, 0] = float("-inf")
            sampling_bounds[:, 1] = float("inf")

            circular_indices_mask = torch.zeros((self.dim,), dtype=torch.bool)
            circular_indices_mask[system.IC_shape_info.circular_indices()] = True

            assert system.spline_range is not None
            sampling_bounds[~circular_indices_mask, 0] = system.spline_range[0]
            sampling_bounds[~circular_indices_mask, 1] = system.spline_range[1]

            if config.AIS_trafo_config is not None:
                assert not any(
                    [item[0] for item in config.AIS_trafo_config.values()]
                ), "Unbounding atanh trafo not supported in combination with rejecting OOB samples. Use only one of the two."

                counter = 0
                for channel, (dims,) in system.IC_shape_info.items():
                    if channel.name in config.AIS_trafo_config:
                        sampling_bounds[
                            counter : counter + dims, :
                        ] *= config.AIS_trafo_config[channel.name][1]
                    counter += dims

            sampling_bounds = sampling_bounds.to("cuda")
        else:
            sampling_bounds = None

        self.base_distribution = BaseDistributionWrapper(
            self.generator_IC,
            system.IC_shape_info,
        )

        self.transition_operator = HamiltonianMonteCarlo(
            n_ais_intermediate_distributions=config.n_int_dist,
            dim=self.dim,
            base_log_prob=self.base_distribution.log_prob,
            target_log_prob=self.target_log_prob_IC,
            p_target=False,
            alpha=config.alpha,
            n_outer=config.n_outer,
            L=config.n_inner,
            epsilon=config.epsilon,
            common_epsilon_init_weight=config.common_epsilon_init_weight,
            sampling_bounds=sampling_bounds,
        )
        self.transition_operator = self.transition_operator.to("cuda")

        self.annealed_importance_sampler = AnnealedImportanceSampler(
            base_distribution=self.base_distribution,
            target_log_prob=self.target_log_prob_IC,
            transition_operator=self.transition_operator,
            n_intermediate_distributions=config.n_int_dist,
            distribution_spacing_type="linear",
            p_target=False,
            alpha=config.alpha,
        )

        # Used to skip counting energy calls for the AIS evaluation
        self._is_evaluating = False

    @property
    def main_temp(self) -> float:
        return self._main_temp

    @property
    def energy_calls_counter(self) -> int:
        """
        NO. target energy calls so far.
        """
        return self._energy_calls_counter

    def save(self, dir_path: str):
        data = {
            "transition_operator": self.transition_operator.state_dict(),  # Save current step size
            "energy_calls_counter": self.energy_calls_counter,
        }
        torch.save(data, os.path.join(dir_path, "ais_base.pickle"))

    def load(self, dir_path: str):
        data = torch.load(os.path.join(dir_path, "ais_base.pickle"))
        self.transition_operator.load_state_dict(data["transition_operator"])
        self._energy_calls_counter = data["energy_calls_counter"]

    def set_ais_target(self, min_is_target: bool = True):
        """Set target to minimum importance sampling distribution
        (p^2/q in case of alpha=2 divergence) for estimating the loss.
        If False, then the AIS target is set to p.

        Args:
            min_is_target: Whether the minimum importance sampling distribution should
                be used as the AIS target. If False, the target is set to p.
        """

        if not min_is_target:
            self.annealed_importance_sampler.p_target = True
            self.annealed_importance_sampler.transition_operator.p_target = True
        else:
            self.annealed_importance_sampler.p_target = False
            self.annealed_importance_sampler.transition_operator.p_target = False

    def target_log_prob_IC(
        self, z: torch.Tensor, return_cart_samples: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        if not self._is_evaluating:
            self._energy_calls_counter += z.shape[0]

        # Undo the concatenation:
        z = torch.split(z, self.dims, dim=1)

        x, log_det_jac = self.IC_trafo.forward(*z)

        target_log_prob = (-self.system.energy(x) + log_det_jac).squeeze()

        if not return_cart_samples:
            return target_log_prob
        else:
            return target_log_prob, x

    def run_additional_sampling_evaluation(
        self,
        is_initial_eval: bool,
        eval_config: EvaluationConfig,
        system: System,
        system_cfg: SystemConfig,
        val_datasets_IC: dict,
        current_i: int,
        marginals_2D: Dict[str, List[Tuple[int, int]]] | None = None,
        marginals_2D_vmax: float | None = 6.5,
        report_wandb: bool = True,
        output_dir: str | None = None,
    ):
        """Run an additional sampling evaluation, e.g. for additional plots or metrics.

        Args:
            is_initial_eval: Whether this is the initial evaluation before training starts.
                This is useful in case you want to skip the initial evaluation (also partially) here.
            eval_config: Evaluation configuration.
            system: The system to evaluate.
            system_cfg: System configuration.
            val_datasets_IC: The validation datasets (IC space).
            current_i: The current iteration in the training loop.
            marginals_2D: The 2D marginal angles to plot (list of tuples of the two torsion indices of the 2Dmarginal).
            marginals_2D_vmax: The vmax for the 2D marginal plots.
            report_wandb: Whether to report to W&B. If False, we print the metrics to the stdout.
            output_dir: The output directory. If None, no output is saved to disk.
        """

        if is_initial_eval:
            return

        self._is_evaluating = True  # Stop counting energy calls for the AIS evaluation

        try:
            eval_batch_size = 4000

            self.transition_operator.set_eval_mode(
                True
            )  # Turn off step size adjustment for AIS eval

            for target_min_distribution in (
                [True, False]
                if self._ais_base_config.additional_sampling_eval.eval_p2q
                else [False]
            ):
                print(
                    "Running AIS sampling evaluation with target_min_distribution =",
                    target_min_distribution,
                )
                start_time = time()

                self.set_ais_target(min_is_target=target_min_distribution)

                n_samples = eval_config.additional_sampling_eval_samples

                flow_samples_ICs = []
                for i in range(len(val_datasets_IC[self.main_temp])):
                    flow_samples_ICs.append(
                        torch.empty(
                            (n_samples, val_datasets_IC[self.main_temp][i].shape[1])
                        )
                    )
                log_weights = torch.empty((n_samples,))

                for i in range(0, n_samples, eval_batch_size):
                    n_samples_batch = min(eval_batch_size, n_samples - i)

                    point_ais, log_w_ais = (
                        self.annealed_importance_sampler.sample_and_log_weights(
                            n_samples_batch, logging=False
                        )
                    )
                    new_samples = self.base_distribution.split_samples(point_ais.x)
                    *new_samples, _ = self.inv_AIS_trafo.forward(*new_samples)
                    *new_samples, _ = self.constrain_trafo.forward(*new_samples)

                    for j in range(len(flow_samples_ICs)):
                        flow_samples_ICs[j][i : i + n_samples_batch] = (
                            new_samples[j].detach().cpu()
                        )
                    log_weights[i : i + n_samples_batch] = log_w_ais.detach().cpu()

                ESS = calculate_reverse_ESS(
                    log_weights=log_weights,
                    clipping_fraction=system_cfg.eval_IS_clipping_fraction,
                )
                if report_wandb:
                    wandb.log(
                        {
                            f"ESS_AIS{'_min' if target_min_distribution else '_p'}": ESS,
                        },
                        step=current_i,
                    )
                else:
                    print(
                        f"ESS_AIS{'_min' if target_min_distribution else '_p'}: {ESS}"
                    )

                if (
                    system_cfg.eval_IS_clipping_fraction is not None
                ):  # Also clip for the marginal plots
                    k = int(log_weights.shape[0] * system_cfg.eval_IS_clipping_fraction)
                    if k > 1:
                        clip_value = torch.min(torch.topk(log_weights, k).values)
                        log_weights[log_weights > clip_value] = clip_value

                log_weights = log_weights - torch.max(log_weights)
                weights = torch.exp(log_weights)

                for i, key in enumerate(system.IC_shape_info.keys()):
                    if key.is_circular:
                        flow_samples_ICs[i] = flow_samples_ICs[i] % 1.0

                ground_truth_ICs = val_datasets_IC[self.main_temp]
                downsampled_ground_truth_ICs = []
                for i in range(len(ground_truth_ICs)):
                    downsampled_ground_truth_ICs.append(ground_truth_ICs[i][:n_samples])

                for flow_data, ground_truth_data, tag, marginals_2D in zip(
                    flow_samples_ICs,
                    downsampled_ground_truth_ICs,
                    [channel_name for channel_name in system.IC_shape_info.names()],
                    [
                        (
                            system_cfg.marginals_2D[channel_name]
                            if channel_name in system_cfg.marginals_2D
                            else None
                        )
                        for channel_name in system.IC_shape_info.names()
                    ],
                ):
                    for weighted in [True, False]:
                        system.plot_IC_marginals(
                            flow_data=flow_data,
                            ground_truth_data=ground_truth_data,
                            tag=f"AIS_{tag}"
                            + ("_min" if target_min_distribution else "_p")
                            + ("_weighted" if weighted else ""),
                            current_i=current_i,
                            flow_data_weights=(weights if weighted else None),
                            plot_as_free_energy=False,
                            marginals_2D=marginals_2D,
                            marginals_2D_vmax=system_cfg.marginals_2D_vmax,
                            report_wandb=report_wandb,
                            output_dir=output_dir,
                            do_calculate_2D_marginal_kld=False,
                            is_big_eval=True,  # We want 1D marginals here.
                        )

                print(
                    f"AIS sampling evaluation with target_min_distribution={target_min_distribution} took {time() - start_time} seconds."
                )

        finally:
            self._is_evaluating = False  # Start counting energy calls again
