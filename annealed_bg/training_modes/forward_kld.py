import torch
from bgflow import BoltzmannGenerator

from annealed_bg.config.training_modes.forward_kld import ForwardKLDTrainingModeConfig
from annealed_bg.training_modes.base import TrainingMode


class ForwardKLDTrainingMode(TrainingMode):
    def __init__(
        self,
        config: ForwardKLDTrainingModeConfig,
        generator: BoltzmannGenerator,
        main_temp: float,
    ):
        self._config = config
        self.generator = generator
        self._main_temp = main_temp

    @property
    def config(self) -> ForwardKLDTrainingModeConfig:
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
        return 0

    @property
    def needs_samples(self) -> bool:
        """
        Whether the training mode needs samples from the target distribution to calculate the loss.
        """
        return True

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

        assert batch is not None

        forward_kls = self.generator.energy(batch, context=None)

        assert len(forward_kls.shape) == 2
        assert forward_kls.shape[1] == 1

        loss = forward_kls.mean()

        return loss, {}
