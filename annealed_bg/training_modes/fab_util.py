from typing import Tuple

import bgflow as bg
import torch
from bgflow import BoltzmannGenerator


class BaseDistributionWrapper:
    def __init__(
        self,
        generator_IC: BoltzmannGenerator,
        shape_info: bg.ShapeDictionary,
    ):
        self.generator_IC = generator_IC
        self.dims = [item[0] for item in shape_info.values()]

    def sample_and_log_prob(self, shape: Tuple):

        assert len(shape) == 1, "Only 1D shapes are supported"
        output = self.generator_IC.sample(shape[0], with_energy=True)

        samples = output[0:-1]
        energies = output[-1]

        # The samples returned by the generator are in form of a tuple, one entry for bonds, angles, and torsions in case of a molecular system
        # Concatenate them into a single tensor:
        samples = self.merge_samples(samples)

        return samples, (-1.0 * energies).squeeze()

    def log_prob(self, z: torch.Tensor):
        # Reverse the concatenation of the samples to obtain the log_prob from the generator:
        z = self.split_samples(z)

        return (-1.0 * self.generator_IC.energy(*z)).squeeze()

    def split_samples(self, x: torch.Tensor):
        return torch.split(x, self.dims, dim=1)

    def merge_samples(self, x: Tuple[torch.Tensor]):
        return torch.cat(x, dim=1)
