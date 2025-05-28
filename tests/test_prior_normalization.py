import bgflow as bg
import numpy as np
import torch
import tqdm
from bgflow import BoltzmannGenerator
from scipy.stats.qmc import Sobol

from tests.utils import create_toy_generator


def test_prior_normalization():
    torch.set_default_dtype(torch.float64)

    # Testing normalization of prior of toy generator with dims=4:
    _test_prior_normalization()


def _test_prior_normalization(
    generator: BoltzmannGenerator = None, shape_info: bg.ShapeDictionary = None
):
    assert (generator is None) == (shape_info is None)

    if generator is None:
        generator, shape_info = create_toy_generator()

    dims = [item[0] for item in shape_info.values()]

    generator = generator.to("cuda")

    min_value = 0.0
    max_value = 1.0

    all_dims = sum(dims)

    batch_size = 2**17
    n_batches = 10000

    integrals = []

    sampler = Sobol(d=all_dims, scramble=True, seed=np.random.randint(0, 2**32 - 1))

    with torch.no_grad():
        for _ in tqdm.tqdm(range(n_batches)):
            if sampler.num_generated + batch_size > sampler.maxn:
                sampler = Sobol(
                    d=all_dims, scramble=True, seed=np.random.randint(0, 2**32 - 1)
                )

            samples = (max_value - min_value) * torch.tensor(
                sampler.random(batch_size), dtype=torch.float32
            ) + min_value
            samples = samples.to("cuda")

            samples = torch.split(samples, dims, dim=1)

            log_probs = -1.0 * generator._prior.energy(*samples)

            log_probs = log_probs.view(-1)

            log_probs_max = torch.max(log_probs)
            integral = (
                torch.mean(torch.exp(log_probs - log_probs_max))
                * (max_value - min_value) ** all_dims
            )
            integral = integral * torch.exp(log_probs_max)

            integrals.append(integral.item())

        integral = np.mean(integrals)

        assert np.abs(integral - 1.0) < 1e-4
