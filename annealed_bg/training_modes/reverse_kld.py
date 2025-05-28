import os

import torch
from bgflow import BoltzmannGenerator

from annealed_bg.config.training_modes.reverse_kld import ReverseKLDTrainingModeConfig
from annealed_bg.training_modes.base import TrainingMode


class ReverseKLDTrainingMode(TrainingMode):
    def __init__(
        self,
        config: ReverseKLDTrainingModeConfig,
        generator: BoltzmannGenerator,
        batch_size: int,
        main_temp: float,
    ):
        self._config = config
        self.generator = generator
        self.batch_size = batch_size
        self._main_temp = main_temp

        self._energy_calls_counter = 0

    @property
    def config(self) -> ReverseKLDTrainingModeConfig:
        """
        Configuration object for the training mode.
        """
        return self._config

    @property
    def main_temp(self) -> float:
        return self._main_temp

    @property
    def energy_calls_counter(self) -> int:
        """
        NO. target energy calls so far.
        """
        return self._energy_calls_counter

    @property
    def needs_samples(self) -> bool:
        """
        Whether the training mode needs samples from the target distribution to calculate the loss.
        """
        return False

    def save(self, dir_path: str):
        others = {
            "energy_calls_counter": self._energy_calls_counter,
        }
        torch.save(others, os.path.join(dir_path, "others.pickle"))

    def load(self, dir_path: str):
        others = torch.load(os.path.join(dir_path, "others.pickle"))
        self._energy_calls_counter = others["energy_calls_counter"]

    def calculate_loss(
        self, current_i: int | None = None, batch: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict]:
        """Calculate the loss for the current iteration.

        Args:
            current_i (int): Current training iteration.
            batch (torch.Tensor, optional): Batch of samples. Defaults to None.

        Returns:
            tuple[torch.Tensor, dict]: Loss value and additional metrics.
        """

        additional_metrics = {}

        self._energy_calls_counter += self.batch_size

        reverse_kls, energies = self.generator.kldiv(
            self.batch_size,
            context=None,
            temperature=self.main_temp,
            return_energies=True,
        )

        assert len(reverse_kls.shape) == 2
        assert reverse_kls.shape[1] == 1
        assert len(energies.shape) == 2
        assert energies.shape[1] == 1

        reverse_kls = reverse_kls.view(-1)
        energies = energies.view(-1)
        k = self.config.remove_top_k_energies

        additional_metrics["train_loss_before_filtering"] = reverse_kls.mean().item()

        if k is not None and k > 0:
            reverse_kls = reverse_kls[torch.argsort(energies, descending=True)[k:]]

        loss = reverse_kls.mean()

        return loss, additional_metrics
