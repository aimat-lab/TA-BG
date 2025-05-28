import mdtraj
import numpy as np
import torch


def all_distances(xs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(xs, np.ndarray):
        mask = np.triu(np.ones([xs.shape[-2], xs.shape[-2]]), k=1).astype(bool)
        xs2 = np.square(xs).sum(axis=-1)
        ds2 = (
            xs2[..., None]
            + xs2[..., None, :]
            - 2 * np.einsum("nid, njd -> nij", xs, xs)
        )
        ds2 = ds2[:, mask].reshape(xs.shape[0], -1)
        ds = np.sqrt(ds2)
    else:
        assert isinstance(xs, torch.Tensor)
        mask = torch.triu(torch.ones([xs.shape[-2], xs.shape[-2]]), diagonal=1).bool()
        xs2 = xs.pow(2).sum(dim=-1)
        ds2 = (
            xs2[..., None]
            + xs2[..., None, :]
            - 2 * torch.einsum("nid, njd -> nij", xs, xs)
        )
        ds2 = ds2[:, mask].view(xs.shape[0], -1)
        ds = ds2.sqrt()
    return ds


def to_tics(
    xs: torch.Tensor,
    mdtraj_topology: mdtraj.Topology,
    tica_file_path: str,
    selection: str = "name == CA",
    eigs_kept: int | None = None,
):
    npz = np.load(tica_file_path)
    tica_mean, tica_eig = npz["tica_mean"], npz["tica_eigenvectors"]

    tica_mean = torch.from_numpy(tica_mean).float()
    tica_eig = torch.from_numpy(tica_eig).float()

    c_alpha = mdtraj_topology.select(selection)
    xs = xs.reshape(xs.shape[0], -1, 3)
    xs = xs[:, c_alpha, :]
    if eigs_kept is None:
        eigs_kept = tica_eig.shape[-1]
    dists = all_distances(xs)
    return (dists - tica_mean) @ tica_eig[:, :eigs_kept]
