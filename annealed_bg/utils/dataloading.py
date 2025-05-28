from typing import Literal

import numpy as np
import torch
from bgflow.nn.flow.sequential import SequentialFlow


def load_trajectory(path: str, data_type: Literal["openmm", "raw"]) -> torch.Tensor:
    data = np.load(path)

    if data_type == "openmm":
        data = data * 0.1  # angstrom to nm

    data = data.reshape(data.shape[0], -1)

    # Convert to torch tensor with the correct precision:
    data = torch.tensor(data, dtype=torch.get_default_dtype())

    return data


def convert_to_IC(
    data_cartesian: torch.Tensor, IC_trafo: SequentialFlow
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    batch_size = 100000

    data_IC = None

    with torch.no_grad():
        for i in range(0, data_cartesian.shape[0], batch_size):
            n_samples_batch = min(batch_size, data_cartesian.shape[0] - i)
            samples = data_cartesian[i : i + n_samples_batch]

            samples_IC = IC_trafo.forward(samples.cuda(), inverse=True)
            samples_IC = [s.cpu() for s in samples_IC[:-1]]  # exclude log_det

            if data_IC is None:
                data_IC = samples_IC
            else:
                for j in range(len(samples_IC)):
                    data_IC[j] = torch.cat((data_IC[j], samples_IC[j]), dim=0)

    return data_IC


# Source of the following dataloader code:
# https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """

    def __init__(
        self,
        *tensors: torch.Tensor,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = True
    ):
        """
        Initialize a FastTensorDataLoader.

        Args:
            *tensors: Tensors to store. Must have the same length @ dim 0.
            batch_size: batch size to load.
            shuffle: If True, shuffle the data *in-place* whenever an
                iterator is created out of this object.
            drop_last: If True, drop the last incomplete batch, if the dataset
                size is not divisible by the batch size. If False and the size of dataset
                is not divisible by the batch size, then the last batch will be smaller.
        """

        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0 and not drop_last:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len or (
            self.drop_last and (self.dataset_len - self.i) < self.batch_size
        ):
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i : self.i + self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
        else:
            batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
