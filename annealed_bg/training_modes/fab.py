import os

import numpy as np
import torch
from bgflow import BoltzmannGenerator
from bgflow.nn.flow.sequential import SequentialFlow
from fab.sampling_methods import Point
from fab.utils.prioritised_replay_buffer import PrioritisedReplayBuffer

from annealed_bg.config.training_modes.fab import FabTrainingModeConfig
from annealed_bg.systems.base import System
from annealed_bg.training_modes.ais_base import AISBaseTrainingMode

##### Parts of this code are adapted from https://github.com/lollcat/fab-torch #####


class FABTrainingMode(AISBaseTrainingMode):
    def __init__(
        self,
        config: FabTrainingModeConfig,
        generator_IC: BoltzmannGenerator,
        IC_trafo: SequentialFlow,
        system: System,
        batch_size: int,
        main_temp: float,
    ):
        AISBaseTrainingMode.__init__(
            self, config, generator_IC, IC_trafo, system, main_temp
        )

        self._config = config
        self.batch_size = batch_size

        if config.replay_buffer is not None:
            rb_config = config.replay_buffer
            self.buffer = PrioritisedReplayBuffer(
                dim=self.dim,
                max_length=rb_config.max_length * batch_size,
                min_sample_length=rb_config.min_length * batch_size,
                initial_sampler=None,
                fill_buffer_during_init=False,  # We do this ourselves once the buffer is needed the first time
                device="cuda",
            )
        else:
            self.buffer = None

        self._buffer_filled = False
        self.buffer_iter = None

    @property
    def config(self) -> FabTrainingModeConfig:
        """
        Configuration object for the training mode.
        """
        return self._config

    @property
    def needs_samples(self) -> bool:
        """
        Whether the training mode needs samples from the target distribution to calculate the loss.
        """
        return False

    def save(self, dir_path: str):
        if self.config.replay_buffer is not None:
            self.buffer.save(os.path.join(dir_path, "buffer.pt"))

        others = {
            "buffer_iter": self.buffer_iter,
            "buffer_filled": self._buffer_filled,
        }
        torch.save(others, os.path.join(dir_path, "others.pickle"))

        AISBaseTrainingMode.save(self, dir_path)

    def load(self, dir_path: str):
        if self.config.replay_buffer is not None:
            self.buffer.load(os.path.join(dir_path, "buffer.pt"))

        others = torch.load(os.path.join(dir_path, "others.pickle"))
        self.buffer_iter = others["buffer_iter"]
        self._buffer_filled = others["buffer_filled"]

        AISBaseTrainingMode.load(self, dir_path)

    def fill_buffer_initially(self):

        assert self.buffer is not None
        print("Initially filling replay buffer...")

        self.set_ais_target(min_is_target=True)
        self.transition_operator.set_eval_mode(
            not self.config.adjust_step_size_training
        )

        # Fill buffer up to minimum length
        while not self.buffer.can_sample:
            point, log_w = self.annealed_importance_sampler.sample_and_log_weights(
                self.batch_size, logging=False
            )
            x, log_w, log_q_old = point.x, log_w, point.log_q
            self.buffer.add(x, log_w, log_q_old)

        self._buffer_filled = True

    def _fab_alpha_div_inner(
        self, point: Point, log_w_ais: torch.Tensor
    ) -> torch.Tensor:
        """Compute FAB loss based off points and importance weights from AIS targetting
        p^\alpha/q^{\alpha-1}.
        """

        log_q_x = self.base_distribution.log_prob(point.x)
        return -np.sign(self.config.alpha) * torch.mean(
            torch.softmax(log_w_ais, dim=-1) * log_q_x
        )

    def _fab_alpha_div_unbuffered(self, batch_size: int):
        point_ais, log_w_ais = self.annealed_importance_sampler.sample_and_log_weights(
            batch_size
        )
        log_w_ais = log_w_ais.detach()
        loss = self._fab_alpha_div_inner(point_ais, log_w_ais)
        return loss

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

        self.set_ais_target(min_is_target=True)
        self.transition_operator.set_eval_mode(
            not self.config.adjust_step_size_training
        )

        if self.config.replay_buffer is not None:
            assert (
                current_i is not None
            ), "Iteration number must be provided when using replay buffer"

            if not self._buffer_filled:
                self.fill_buffer_initially()

            if current_i % self.config.replay_buffer.n_updates == 0:
                # Sample
                point_ais, log_w_ais = (
                    self.annealed_importance_sampler.sample_and_log_weights(
                        self.batch_size, logging=False
                    )
                )

                # Add sample to buffer
                self.buffer.add(point_ais.x, log_w_ais.detach(), point_ais.log_q)

                # Sample from buffer
                buffer_sample = self.buffer.sample_n_batches(
                    batch_size=self.batch_size,
                    n_batches=self.config.replay_buffer.n_updates,
                )
                self.buffer_iter = iter(buffer_sample)

            # Get batch from buffer
            x, log_w, log_q_old, indices = next(self.buffer_iter)
            x, log_w, log_q_old, indices = (
                x.to("cuda"),
                log_w.to("cuda"),
                log_q_old.to("cuda"),
                indices.to("cuda"),
            )
            log_q_x = self.base_distribution.log_prob(x)

            # Adjustment to account for change to theta since sample was last added/adjusted
            log_w_adjust = (1 - self.config.alpha) * (log_q_x.detach() - log_q_old)
            w_adjust = torch.clip(
                torch.exp(log_w_adjust),
                max=self.config.replay_buffer.max_adjust_w_clip,
            )

            # Manually calculate the new form of the loss
            loss = -torch.mean(w_adjust * log_q_x)

            # Adjust buffer samples
            self.buffer.adjust(log_w_adjust, log_q_x.detach(), indices)

        else:
            loss = self._fab_alpha_div_unbuffered(self.batch_size)

        return loss, {}
