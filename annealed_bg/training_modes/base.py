from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch

from annealed_bg.config.evaluation import EvaluationConfig
from annealed_bg.config.system import SystemConfig
from annealed_bg.config.training_modes.base import TrainingModeConfig
from annealed_bg.systems.base import System


class TrainingMode(ABC):
    """
    Abstract base class for concrete loss implementations.
    """

    @property
    def state_to_log(self) -> dict:
        """
        Additional state variables to log to wandb.
        """
        return {}

    @property
    @abstractmethod
    def config(self) -> TrainingModeConfig:
        """
        Configuration object for the training mode.
        """
        pass

    @property
    @abstractmethod
    def main_temp(self) -> float:
        pass

    @property
    @abstractmethod
    def energy_calls_counter(self) -> int:
        """
        NO. target energy calls so far.
        """
        pass

    @property
    @abstractmethod
    def needs_samples(self) -> bool:
        """
        Whether the training mode needs samples from the target distribution to calculate the loss.
        """
        pass

    @property
    def only_determine_permutation_once(self) -> bool:
        """
        If config.evaluation.apply_cart_permutation_to_ground_truth_datasets is True, this flag
        determines whether the permutation should be determined only once in the beginning of training.
        """
        return False

    @property
    def eval_sampling_T_pairs(self) -> List[Tuple[float | None, float | None]]:
        """
        Default temperature pairs to evaluate when sampling from the generator.
        List of (temperature to sample at (None means no temperature-conditioning), temperature to reweight to (None means no reweighting)).
        """
        return [(None, None), (None, self.main_temp)]

    @property
    def eval_NLL_Ts(self) -> List[float]:
        """
        Temperatures to evaluate the NLL at.
        """
        return [self.main_temp]

    def save(self, dir_path: str):
        """
        Save the current state to a directory.
        """
        pass  # By default, do nothing

    def load(self, dir_path: str):
        """
        Load the state from a directory.
        """
        pass  # By default, do nothing

    @abstractmethod
    def calculate_loss(
        self, current_i: int, batch: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict]:
        """Calculate the loss for the current iteration.

        Args:
            current_i: Current training iteration.
            batch: Batch of samples.

        Returns:
            Loss value and additional metrics.
        """
        pass

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
        pass
