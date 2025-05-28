import gc
import os
import time
import traceback
from typing import List, Tuple

import numpy as np
import torch
import wandb
from bgflow import BoltzmannGenerator

from annealed_bg.config.training_modes.reweighting import (
    ContinuousTempSamplingConfig,
    ReweightingTrainingModeConfig,
    SequenceTempSamplingConfig,
)
from annealed_bg.evaluation.metrics import calculate_reverse_ESS
from annealed_bg.training_modes.base import TrainingMode
from annealed_bg.utils.rounded_dict import RoundedKeyDict


class ReweightingTrainingMode(TrainingMode):
    def __init__(
        self,
        config: ReweightingTrainingModeConfig,
        generator: BoltzmannGenerator,
        batch_size: int,
        reinit_fn: callable,
        optimizer: torch.optim.Optimizer,
        main_temp: float,
        cart_dims: int,
        eval_IS_clipping_fraction: float = 1.0e-4,
    ):
        self._energy_calls_counter = 0

        self.generator = generator
        self.optimizer = optimizer

        self.reinit_fn = reinit_fn

        self._config = config

        self.batch_size = batch_size

        self.cart_dims = cart_dims

        self._main_temp = main_temp

        self.current_temperature_range = None

        self._eval_IS_clipping_fraction = eval_IS_clipping_fraction

        if isinstance(self.config.temp_sampling, SequenceTempSamplingConfig):
            self.sequence = self.config.temp_sampling.sequence

            print(f"Reweighting sequence: {self.sequence}")

            self.reweight_temps = torch.tensor([x[1] for x in self.sequence])
            print(f"Reweighting temperatures in sequence: {self.reweight_temps}")
            self.reweight_temps = self.reweight_temps.to("cuda")

            self.cum_iterations = []
            total_iterations = 0
            for item in self.sequence:
                if len(item) > 2:
                    iterations_per_step = item[2]
                else:
                    iterations_per_step = (
                        self.config.temp_sampling.iterations_per_step
                    )  # Use default value

                assert iterations_per_step is not None

                if len(item) > 3:
                    reinit_generator = item[3]

                    if reinit_generator:
                        assert self.config.buffer.update_buffer_every is None

                total_iterations += iterations_per_step
                self.cum_iterations.append(total_iterations)

        if self.config.buffer is not None:
            self._buffer_samples_per_T = RoundedKeyDict(digits=2)
            self._buffer_weights_per_T = RoundedKeyDict(digits=2)
            self._buffer_counter_per_T = RoundedKeyDict(digits=2)

            for T in self.reweight_temps:
                self._buffer_samples_per_T[T.item()] = None
                self._buffer_weights_per_T[T.item()] = None
                self._buffer_counter_per_T[T.item()] = 0

        else:
            self._buffer_samples_per_T = None
            self._buffer_weights_per_T = None
            self._buffer_counter_per_T = None

    @property
    def config(self) -> ReweightingTrainingModeConfig:
        """
        Configuration object for the training mode.
        """
        return self._config

    @property
    def main_temp(self) -> float:
        return self._main_temp

    @property
    def state_to_log(self) -> dict:
        """
        Additional state variables to log to wandb.
        """
        if self.current_temperature_range is None:
            return {}
        else:
            return {"current_temperature_range_left": self.current_temperature_range[0]}

    @property
    def energy_calls_counter(self) -> int:
        """
        NO. target energy calls so far.
        """
        return self._energy_calls_counter

    @property
    def needs_samples(self):
        """
        Whether the training mode needs samples from the target distribution to calculate the loss.
        """
        return False

    @property
    def only_determine_permutation_once(self) -> bool:
        """
        If config.evaluation.apply_cart_permutation_to_ground_truth_datasets is True, this flag
        determines whether the permutation should be determined only once in the beginning of training.
        """
        # We usually use a pretrained checkpoint at "main_temp" when running in the reweighting training mode.
        # Therefore, the permutation will not change during training.
        return True

    @property
    def eval_sampling_T_pairs(self) -> List[Tuple[float | None, float | None]]:
        """
        Default temperature pairs to evaluate when sampling from the generator.
        List of (temperature to sample at (None means no temperature-conditioning), temperature to reweight to (None means no reweighting)).
        """
        return [
            (None, None),
            (self.config.max_temperature_range[0], None),
            (
                None,
                self.config.max_temperature_range[0],
            ),  # Reweighting to the lowest temperature
        ]

    @property
    def eval_NLL_Ts(self) -> List[float]:
        """
        Temperatures to evaluate the NLL at.
        """
        return [self.main_temp, self.config.max_temperature_range[0]]

    def save(self, dir_path: str):
        if self.config.buffer is not None:
            torch.save(
                {
                    "buffer_samples_per_T": (
                        self._buffer_samples_per_T.to_dict()
                        if self._buffer_samples_per_T is not None
                        else None
                    ),
                    "buffer_weights_per_T": (
                        self._buffer_weights_per_T.to_dict()
                        if self._buffer_weights_per_T is not None
                        else None
                    ),
                    "buffer_counter_per_T": (
                        self._buffer_counter_per_T.to_dict()
                        if self._buffer_counter_per_T is not None
                        else None
                    ),
                },
                os.path.join(dir_path, "buffer_reweighting.pickle"),
            )

        torch.save(
            {
                "energy_calls_counter": self._energy_calls_counter,
            },
            os.path.join(dir_path, "others_reweighting.pickle"),
        )

    def load(self, dir_path: str):
        if self.config.buffer is not None:
            buffer_dict = torch.load(
                os.path.join(dir_path, "buffer_reweighting.pickle")
            )

            self._buffer_samples_per_T = (
                RoundedKeyDict(digits=2)
                if buffer_dict["buffer_samples_per_T"] is not None
                else None
            )
            if self._buffer_samples_per_T is not None:
                self._buffer_samples_per_T.fill_with_dict(
                    buffer_dict["buffer_samples_per_T"]
                )

            self._buffer_weights_per_T = (
                RoundedKeyDict(digits=2)
                if buffer_dict["buffer_weights_per_T"] is not None
                else None
            )
            if self._buffer_weights_per_T is not None:
                self._buffer_weights_per_T.fill_with_dict(
                    buffer_dict["buffer_weights_per_T"]
                )

            self._buffer_counter_per_T = (
                RoundedKeyDict(digits=2)
                if buffer_dict["buffer_counter_per_T"] is not None
                else None
            )
            if self._buffer_counter_per_T is not None:
                self._buffer_counter_per_T.fill_with_dict(
                    buffer_dict["buffer_counter_per_T"]
                )

        others = torch.load(os.path.join(dir_path, "others_reweighting.pickle"))
        self._energy_calls_counter = others["energy_calls_counter"]

    def _fill_buffer(
        self,
        reweighting_T: float,
        sampling_T: float,
        n_samples_per_T: int,
        resample_to: int | None = None,
    ) -> float:
        assert self._buffer_samples_per_T is not None
        assert self._buffer_weights_per_T is not None
        assert self._buffer_counter_per_T is not None

        clip_top_k_weights_fraction = self.config.buffer.clip_top_k_weights_fraction

        batch_size_filling = int(2**12)

        all_samples = torch.empty((n_samples_per_T, self.cart_dims))
        log_weights = torch.empty((n_samples_per_T, 1))

        start = time.time()
        with torch.no_grad():
            buffer_filling_additional_energy_calls = 0

            for i in range(0, n_samples_per_T, batch_size_filling):
                n_samples_batch = min(batch_size_filling, n_samples_per_T - i)

                context = torch.ones(n_samples_batch, 1, device="cuda") * sampling_T

                samples, bg_energies = self.generator.sample(
                    n_samples_batch, with_energy=True, context=context
                )
                log_qs = -bg_energies

                buffer_filling_additional_energy_calls += samples.size(0)

                # Target energies at the reweighting temperature:
                energies_reweighting_T = self.generator._target.energy(
                    samples, temperature=reweighting_T
                )

                all_samples[i : i + n_samples_batch] = samples.cpu()
                log_weights[i : i + n_samples_batch] = (
                    -log_qs - energies_reweighting_T
                ).cpu()

            # Filter potential NaNs from log_weights:
            nan_mask = torch.isnan(log_weights.view(-1))
            if torch.any(nan_mask):
                print(
                    "Warning: Found NaNs in log_weights when filling buffer. Number of NaNs:",
                    torch.sum(nan_mask),
                )
                log_weights = log_weights[~nan_mask]
                all_samples = all_samples[~nan_mask]

            # Calculate the ESS of the buffer dataset:
            ESS = calculate_reverse_ESS(
                log_weights=log_weights.view(-1),
                clipping_fraction=self._eval_IS_clipping_fraction,
            )

            if (
                clip_top_k_weights_fraction is not None
                and clip_top_k_weights_fraction > 0.0
            ):
                clip_top_k_weights = int(
                    clip_top_k_weights_fraction * log_weights.size(0)
                )
                clip_value = torch.min(
                    torch.topk(log_weights.view(-1), clip_top_k_weights).values
                )
                log_weights[log_weights > clip_value] = clip_value

            max_of_log_weights = torch.max(log_weights)
            weights = torch.exp(log_weights - max_of_log_weights)
            weights /= torch.sum(weights)  # Self-normalize

            assert len(weights.shape) == 2
            assert weights.shape[1] == 1

            if resample_to is None:
                self._buffer_weights_per_T[reweighting_T] = weights
                self._buffer_samples_per_T[reweighting_T] = all_samples
            else:
                # Switch to numpy to avoid 2^24 elements limit of torch.multinomial:
                weights_np = np.float64(weights.view(-1).numpy())

                weights_np /= np.sum(weights_np)

                indices = np.random.choice(
                    np.arange(weights.size(0)),
                    size=resample_to,
                    replace=True,
                    p=weights_np,
                )
                indices = torch.from_numpy(indices)

                self._buffer_samples_per_T[reweighting_T] = all_samples[indices]
                self._buffer_weights_per_T[reweighting_T] = None  # No weights needed

                NO_unique_samples = len(torch.unique(indices))

            self._buffer_counter_per_T[reweighting_T] = 0

            self._energy_calls_counter += buffer_filling_additional_energy_calls

            print(
                "Filled buffer for T =",
                reweighting_T,
                "with",
                self._buffer_samples_per_T[reweighting_T].size(0),
                "samples in",
                time.time() - start,
                "seconds."
                + (
                    ("\n" + str(NO_unique_samples) + " unique samples.")
                    if resample_to is not None
                    else ""
                )
                + f"\n(ESS: {ESS})",
            )

        gc.collect()
        torch.cuda.empty_cache()

        return ESS

    def _get_batch_from_buffer(
        self, reweight_temps_context: torch.Tensor, resample_to: int | None
    ):

        if not isinstance(reweight_temps_context, torch.Tensor):
            raise TypeError("reweight_temps_context must be a torch.Tensor")

        reweight_temps_context = reweight_temps_context.squeeze()
        unique_temps = torch.unique(reweight_temps_context)

        # Ensure all temperatures are in the buffer
        for temp in unique_temps:
            temp = temp.item()
            if temp not in self._buffer_samples_per_T:
                raise ValueError(f"Temperature {temp} not found in the buffer.")

        # Prepare empty batch:
        batch_size = reweight_temps_context.size(0)
        samples = torch.empty((batch_size, self.cart_dims))
        weights = torch.empty((batch_size, 1)) if resample_to is None else None

        # Retrieve samples from the buffer for each unique temperature
        for temp in unique_temps:
            temp = temp.item()
            buffer_samples = self._buffer_samples_per_T[temp]
            buffer_weights = self._buffer_weights_per_T[temp]
            counter = self._buffer_counter_per_T[temp]

            # Determine the indices of this temperature in the input tensor
            indices = (reweight_temps_context == temp).nonzero(as_tuple=True)[0]
            count = len(indices)

            # Check if there are enough samples left, shuffle and reset counter if necessary
            if counter + count > buffer_samples.size(0):
                perm = torch.randperm(buffer_samples.size(0))

                buffer_samples = buffer_samples[perm]
                if buffer_weights is not None:
                    buffer_weights = buffer_weights[perm]

                self._buffer_samples_per_T[temp] = buffer_samples
                self._buffer_weights_per_T[temp] = buffer_weights

                self._buffer_counter_per_T[temp] = 0
                counter = 0

                gc.collect()
                torch.cuda.empty_cache()

            # Retrieve samples and place them in the batch
            samples[indices] = buffer_samples[counter : counter + count]
            if weights is not None:
                weights[indices] = buffer_weights[counter : counter + count]

            self._buffer_counter_per_T[temp] = counter + count

        return samples, weights

    def _get_unbuffered_loss(
        self,
        support_temps_context: torch.Tensor,
        reweight_temps_context: torch.Tensor,
    ):
        batch_size = support_temps_context.size(0)

        # Sample at the support temperatures:
        support_samples, support_bg_energies = self.generator.sample(
            batch_size,
            context=support_temps_context,
            with_energy=True,
        )

        support_samples, support_log_qs = (
            support_samples.detach(),
            -support_bg_energies.detach(),
        )

        self._energy_calls_counter += support_samples.size(0)
        log_ps_at_reweight_temps = -1.0 * self.generator._target.energy(
            support_samples, temperature=reweight_temps_context
        )

        log_weights = log_ps_at_reweight_temps - support_log_qs

        if (
            self.config.clip_top_k_weights is not None
            and self.config.clip_top_k_weights > 0
        ):
            clip_value = torch.min(
                torch.topk(log_weights.view(-1), self.config.clip_top_k_weights).values
            )
            log_weights[log_weights > clip_value] = clip_value

        if self.config.self_normalize_weights:
            max_of_log_weights = torch.max(log_weights)
            weights = torch.exp(log_weights - max_of_log_weights)
            weights = weights / torch.sum(weights)  # Self-normalize
        else:
            fix_max_value = -52.0
            weights = torch.exp(log_weights - fix_max_value)
            weights *= 10.0

        weights = weights.detach()

        assert len(weights.shape) == 2
        assert weights.shape[1] == 1

        # TODO: Can be sped up in case of resampling
        log_qs_at_reweight_temps = -1.0 * self.generator.energy(
            support_samples, context=reweight_temps_context
        )

        if self.config.resample_batch_to is None:
            temp_scaling_losses = -1.0 * (weights * log_qs_at_reweight_temps)
        else:
            indices = torch.multinomial(
                weights.view(-1),
                self.config.resample_batch_to,
                replacement=True,
            )
            temp_scaling_losses = -1.0 * log_qs_at_reweight_temps[indices]

        assert len(temp_scaling_losses.shape) == 2
        assert temp_scaling_losses.shape[1] == 1

        temp_scaling_loss = temp_scaling_losses.mean()

        return temp_scaling_loss

    def _continuous_get_support_and_reweight_temps(self, it: int):

        assert isinstance(self.config.temp_sampling, ContinuousTempSamplingConfig)

        ##### Determine the current temperature range #####
        if self.config.temp_sampling.grow_range is None:
            current_temperature_range = self.config.max_temperature_range
        else:
            min_step = self.config.temp_sampling.grow_range.min_step
            max_step = self.config.temp_sampling.grow_range.max_step

            if self.config.temp_sampling.min_temperature_range is None:
                min_temperature_range = (
                    self.config.max_temperature_range[1],
                    self.config.max_temperature_range[1],
                )
            else:
                min_temperature_range = self.config.temp_sampling.min_temperature_range

            if it < min_step:
                current_temperature_range = min_temperature_range
            else:
                current_temperature_range = (
                    min_temperature_range[0]
                    - (min_temperature_range[0] - self.config.max_temperature_range[0])
                    * min(
                        (it - min_step) / (max_step - min_step),
                        1.0,
                    ),
                    min_temperature_range[1],
                )

        ##### Sample the support temperatures from this range #####

        if (
            self.config.temp_sampling.support_T_distribution.fraction_at_boundary_T_1_before_iteration
            is not None
            and it
            < self.config.temp_sampling.support_T_distribution.fraction_at_boundary_T_1_before_iteration
        ):
            fraction_at_boundary_T = 1.0
        else:
            fraction_at_boundary_T = (
                self.config.temp_sampling.support_T_distribution.fraction_at_boundary_T
            )

        if self.config.temp_sampling.support_T_distribution.dist_type == "uniform":
            if not self.config.temp_sampling.support_T_distribution.one_T_per_batch:
                support_temps_context = (
                    torch.rand(
                        self.batch_size,
                        1,
                        device="cuda",
                    )
                    * (current_temperature_range[1] - current_temperature_range[0])
                    + current_temperature_range[0]
                )

                if fraction_at_boundary_T is not None:
                    # Overwrite a fraction of the support temperatures with the boundary temperature:
                    num_boundary_T = int(fraction_at_boundary_T * self.batch_size)
                    support_temps_context[:num_boundary_T] = self.main_temp

            else:  # Sample only one support temperature and use it for the whole batch:
                support_temps_context = torch.ones(
                    self.batch_size, 1, device="cuda"
                ) * (
                    torch.rand((), device="cuda")
                    * (current_temperature_range[1] - current_temperature_range[0])
                    + current_temperature_range[0]
                )

                if fraction_at_boundary_T is not None:
                    if torch.rand(()) < fraction_at_boundary_T:
                        support_temps_context = (
                            torch.ones(self.batch_size, 1, device="cuda")
                            * self.main_temp
                        )
        elif (
            self.config.temp_sampling.support_T_distribution.dist_type
            == "left_bound_of_current_range"
        ):
            assert fraction_at_boundary_T is None

            support_temps_context = (
                torch.ones(self.batch_size, 1, device="cuda")
                * current_temperature_range[0]
            )

        else:
            raise ValueError(
                f"Support temperature distribution type {self.config.temp_sampling.support_T_distribution.dist_type} not supported."
            )

        ##### Determine the temperatures we want to reweight to #####

        max_delta_T = self.config.temp_sampling.reweight_T_distribution.max_delta_T
        distribution_type = (
            self.config.temp_sampling.reweight_T_distribution.dist_type
        )  # Either "delta_uniform", "delta_uniform_left", or "delta_constant_left"

        # When determining the delta_Ts, we need to make sure that the temperatures are only sampled in a
        # uniform distribution up to the boundaries of the current temperature range!
        if distribution_type == "delta_uniform_left":
            min_Ts = torch.maximum(
                torch.full_like(
                    support_temps_context,
                    current_temperature_range[0],
                ),
                support_temps_context - max_delta_T,
            )
            max_Ts = support_temps_context
        elif distribution_type == "delta_uniform":
            min_Ts = torch.maximum(
                torch.full_like(
                    support_temps_context,
                    current_temperature_range[0],
                ),
                support_temps_context - max_delta_T,
            )
            max_Ts = torch.minimum(
                torch.full_like(
                    support_temps_context,
                    current_temperature_range[1],
                ),
                support_temps_context + max_delta_T,
            )
        elif distribution_type == "delta_constant_left":
            min_Ts = torch.maximum(
                torch.full_like(
                    support_temps_context,
                    current_temperature_range[0],
                ),
                support_temps_context - max_delta_T,
            )
            max_Ts = min_Ts

        else:
            raise ValueError(
                f"Delta T distribution type {distribution_type} not supported."
            )

        if not self.config.temp_sampling.reweight_T_distribution.one_T_per_batch:
            reweight_temps_context = (
                torch.rand(
                    self.batch_size,
                    1,
                    device="cuda",
                )
                * (max_Ts - min_Ts)
                + min_Ts
            )
        else:  # Sample only one reweight temperature and use it for the whole batch:
            assert (
                self.config.temp_sampling.support_T_distribution.one_T_per_batch
            )  # Otherwise this does not make much sense

            reweight_temps_context = torch.ones(self.batch_size, 1, device="cuda") * (
                torch.rand((), device="cuda") * (max_Ts - min_Ts) + min_Ts
            )

        assert len(reweight_temps_context.shape) == 2
        assert reweight_temps_context.shape[1] == 1

        self.current_temperature_range = current_temperature_range

        return (
            support_temps_context,
            reweight_temps_context,
        )

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

        if isinstance(self.config.temp_sampling, ContinuousTempSamplingConfig):
            support_temps_context, reweight_temps_context = (
                self._continuous_get_support_and_reweight_temps(current_i)
            )

            unbuffered_loss = self._get_unbuffered_loss(
                support_temps_context=support_temps_context,
                reweight_temps_context=reweight_temps_context,
            )

            return unbuffered_loss, {}

        elif isinstance(self.config.temp_sampling, SequenceTempSamplingConfig):
            current_step = 0
            for cum_iterations in self.cum_iterations:
                if current_i < cum_iterations:
                    break
                current_step += 1

            if current_step >= len(self.sequence):
                current_step = len(self.sequence) - 1

                # Only show this warning once:
                if current_i == cum_iterations:
                    print(
                        "Warning: Calculated step exceeds sequence length. Continuing with last sequence step."
                    )

            current_sampling_T = self.sequence[current_step][0]
            current_reweighting_T = self.sequence[current_step][1]

            if len(self.sequence[current_step]) > 3:
                reinit_generator = self.sequence[current_step][3]
            else:
                reinit_generator = False

            if len(self.sequence[current_step]) > 4:
                n_samples = self.sequence[current_step][4]
            else:
                n_samples = self.config.buffer.buffer_n_samples_per_T

            assert n_samples is not None

            if len(self.sequence[current_step]) > 5:
                resample_to = self.sequence[current_step][5]
            else:
                resample_to = self.config.buffer.resample_to

            if len(self.sequence[current_step]) > 6:
                new_lr = self.sequence[current_step][6]
            else:
                new_lr = None

            # Check if we just began a new step in the sequence:
            if (current_step == 0 and current_i == 0) or (
                current_i == self.cum_iterations[current_step - 1]
            ):
                print(
                    "Now reweighting from",
                    current_sampling_T,
                    "to",
                    current_reweighting_T,
                )

            if self.config.buffer is not None and (
                current_i > self.config.buffer.activate_buffer_after
            ):
                if (
                    (
                        self.config.buffer.update_buffer_every is None
                        and (
                            (current_step == 0 and current_i == 0)
                            or (current_i == self.cum_iterations[current_step - 1])
                        )
                    )  # Just began a new step
                    or (
                        (self.config.buffer.update_buffer_every is not None)
                        and current_i % self.config.buffer.update_buffer_every == 0
                    )  # Update frequency specified
                    or (
                        current_i == self.config.buffer.activate_buffer_after + 1
                    )  # Just after activation
                ):
                    print(
                        "Filling buffer,",
                        f"{n_samples} samples at T=",
                        current_sampling_T,
                        "reweighting to T=",
                        current_reweighting_T,
                        f"(resample_to={resample_to})",
                    )

                    tries_counter = 0
                    while tries_counter < 10:
                        try:
                            ESS = self._fill_buffer(
                                reweighting_T=current_reweighting_T,
                                sampling_T=current_sampling_T,
                                n_samples_per_T=n_samples,
                                resample_to=resample_to,
                            )
                            wandb.log(
                                {
                                    "buffer_ESS": ESS,
                                },
                                step=current_i,
                            )
                            break
                        except Exception as e:
                            print("Error while filling buffer:")
                            # Print the full stacktrace:
                            print(traceback.format_exc())
                            print("Retrying...")

                            tries_counter += 1

                    if tries_counter == 10:
                        raise RuntimeError(
                            "Failed to fill buffer after 10 tries. Exiting now."
                        )

                    # Let's clear the buffer at the previous reweighting temperature to save memory:
                    if current_step > 0:
                        previous_reweighting_T = self.sequence[current_step - 1][1]

                        # Check that the previous reweighting T is not the same as the current one:
                        if round(float(previous_reweighting_T), 2) != round(
                            float(current_reweighting_T), 2
                        ):
                            if previous_reweighting_T in self._buffer_samples_per_T:
                                del self._buffer_samples_per_T[previous_reweighting_T]
                            if previous_reweighting_T in self._buffer_weights_per_T:
                                del self._buffer_weights_per_T[previous_reweighting_T]
                            if previous_reweighting_T in self._buffer_counter_per_T:
                                del self._buffer_counter_per_T[previous_reweighting_T]
                            self._buffer_samples_per_T[previous_reweighting_T] = None
                            self._buffer_weights_per_T[previous_reweighting_T] = None
                            self._buffer_counter_per_T[previous_reweighting_T] = 0

                            gc.collect()
                            torch.cuda.empty_cache()

                    if reinit_generator and (
                        (current_step == 0 and current_i == 0)
                        or (current_i == self.cum_iterations[current_step - 1])
                    ):
                        print("Reinitializing generator and optimizer.")

                        assert (
                            self.reinit_fn is not None
                        ), "No reinit_fn provided. Cannot reset generator and optimizer."

                        self.reinit_fn()

                    if new_lr is not None and (
                        (current_step == 0 and current_i == 0)
                        or (current_i == self.cum_iterations[current_step - 1])
                    ):
                        print("Setting lr now to", new_lr)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = new_lr

                else:
                    if (current_step == 0 and current_i == 0) or (
                        current_i == self.cum_iterations[current_step - 1]
                    ):
                        print(
                            "Warning: Began new step in the sequence, but buffer was not updated. This is potentially unwanted behavior."
                        )

            reweight_temps_context = (
                torch.ones(self.batch_size, 1, device="cuda") * current_reweighting_T
            )
            support_temps_context = (
                torch.ones(self.batch_size, 1, device="cuda") * current_sampling_T
            )

            if self.config.buffer is None or (
                self.config.buffer is not None
                and current_i <= self.config.buffer.activate_buffer_after
            ):
                return (
                    self._get_unbuffered_loss(
                        support_temps_context=support_temps_context,
                        reweight_temps_context=reweight_temps_context,
                    ),
                    {},
                )

            else:

                samples, weights = self._get_batch_from_buffer(
                    reweight_temps_context, resample_to=resample_to
                )

                samples = samples.to("cuda")

                if weights is None:  # We resampled
                    loss = self.generator.energy(
                        samples, context=reweight_temps_context
                    ).mean()
                else:
                    weights = weights.to("cuda")
                    loss = torch.mean(
                        self.generator.energy(samples, context=reweight_temps_context)
                        * weights
                    )

                return loss, {}
