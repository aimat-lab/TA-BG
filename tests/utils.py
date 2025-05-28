import os

import bgflow as bg
import hydra
import omegaconf
import torch
from bgflow import ANGLES, BONDS, TORSIONS
from bgmol.zmatrix import ZMatrixFactory, build_fake_topology

from annealed_bg.config.main import Config
from annealed_bg.generator.create_generator import create_generator
from annealed_bg.systems.molecular.system import MolecularSystem


def create_toy_generator(dims: int = 4):
    """Create a toy generator with a simple "fake" topology.

    Args:
        dims: The number of atoms in the topology.
    """

    top, _ = build_fake_topology(dims)
    factory = ZMatrixFactory(top)
    z, fixed = factory.build_naive()

    coordinate_trafo_openmm = bg.GlobalInternalCoordinateTransformation(
        z_matrix=z,
        enforce_boundaries=True,
        normalize_angles=True,
    )

    shape_info = bg.ShapeDictionary.from_coordinate_transform(
        coordinate_trafo_openmm,
    )

    transformer_kwargs = dict()
    transformer_kwargs["spline_disable_identity_transform"] = True

    prior_type = dict()
    prior_kwargs = dict()

    # Only torsions keep the default, which is a [0, 1] uniform distribution
    prior_type[BONDS] = bg.TruncatedNormalDistribution
    prior_type[ANGLES] = bg.TruncatedNormalDistribution
    prior_kwargs[BONDS] = {
        "mu": 0.5,
        "sigma": 0.1,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
    }
    prior_kwargs[ANGLES] = {
        "mu": 0.5,
        "sigma": 0.1,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
    }

    builder = bg.BoltzmannGeneratorBuilder(
        prior_dims=shape_info,
        prior_type=prior_type,
        prior_kwargs=prior_kwargs,
        device="cpu",
        dtype=torch.float32,
    )

    builder.add_condition(
        TORSIONS, ANGLES, init_identity=False, transformer_kwargs=transformer_kwargs
    )
    builder.add_condition(
        ANGLES, BONDS, init_identity=False, transformer_kwargs=transformer_kwargs
    )

    builder.add_map_to_cartesian(coordinate_transform=coordinate_trafo_openmm)
    generator = builder.build_generator()

    return generator, shape_info


def create_molecular_generator(system_name: str = "aldp"):
    """Create a generator for the given system.

    Args:
        system_name: The name of the system to create the generator for.
    """

    with hydra.initialize(config_path="../annealed_bg/configs"):
        cfg = hydra.compose(
            config_name="reverse_kl",
            overrides=[f"+system={system_name}", "main_temp=300.0"],
        )
        cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        cfg: Config = Config(**cfg)

        os.chdir("annealed_bg")

        (
            generator,
            generator_IC,
            IC_trafo,
            system,
        ) = create_generator(
            cfg=cfg,
        )

        assert isinstance(system, MolecularSystem)

        os.chdir("..")

        return generator, system.IC_shape_info, system.openmm_system
