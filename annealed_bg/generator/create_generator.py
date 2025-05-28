import os

import bgflow as bg
import bgmol
import numpy as np
import torch
from bgflow import BoltzmannGenerator
from bgflow.nn.flow.sequential import SequentialFlow
from bgmol import bond_constraints
from openmm import Platform
from openmm.unit import nanometer

from annealed_bg.conditioners.dense import _make_dense_conditioner
from annealed_bg.config.main import Config
from annealed_bg.config.training_modes.reweighting import ReweightingTrainingModeConfig
from annealed_bg.generator.ic_scaler import ICScaler
from annealed_bg.systems.base import System
from annealed_bg.systems.factory import create_system
from annealed_bg.systems.molecular.system import MolecularSystem
from annealed_bg.utils.architecture_parsing import parse_architecture_layer_str
from annealed_bg.utils.permutations import (
    get_torsion_indices_for_permutation_constraints,
)


def create_generator(cfg: Config) -> tuple[
    BoltzmannGenerator,
    BoltzmannGenerator,
    SequentialFlow,
    System,
]:
    system = create_system(
        cfg.system.name,
        energy_regularizer_cfg=cfg.training.energy_regularizer,
        n_workers=cfg.general.n_workers,
        system_temp=cfg.main_temp,
    )
    prior_type, prior_kwargs = system.get_prior_type_and_kwargs(
        cfg.flow.couplings_transform_type
    )
    shape_info = system.IC_shape_info

    ##### Build generator #####

    builder = bg.BoltzmannGeneratorBuilder(
        prior_dims=shape_info,
        prior_type=prior_type,
        prior_kwargs=prior_kwargs,
        target=system,
        device="cuda",
        dtype=torch.float32,
    )

    if cfg.flow.couplings_transform_type == "rnvp":
        transformer_type = bg.AffineTransformer
    elif cfg.flow.couplings_transform_type == "spline":
        transformer_type = bg.ConditionalSplineTransformer
    else:
        raise ValueError("Invalid couplings_transform_type")

    transformer_kwargs = dict()
    if cfg.flow.couplings_transform_type == "spline":
        transformer_kwargs["spline_disable_identity_transform"] = True
        # Note: Settings this to True does not actually disable the identity transform, rather it makes the spline
        # an identity transform if the conditioner network outputs np.log(np.exp(1 - DEFAULT_MIN_DERIVATIVE) - 1),
        # which is what we use for initialization of the bias of the conditioner network.

    if cfg.flow.couplings_transform_type == "spline":
        if system.spline_range is not None:
            transformer_kwargs["bottom"] = system.spline_range[0]
            transformer_kwargs["left"] = system.spline_range[0]
            transformer_kwargs["right"] = system.spline_range[1]
            transformer_kwargs["top"] = system.spline_range[1]

    conditioner_factory = _make_dense_conditioner
    conditioner_kwargs = {}
    conditioner_kwargs["hidden"] = cfg.flow.hidden
    if cfg.flow.use_silu_activation:
        conditioner_kwargs["activation"] = torch.nn.SiLU()
    conditioner_kwargs["add_skip_connection"] = cfg.flow.add_skip_connection
    if cfg.flow.couplings_transform_type == "rnvp":
        conditioner_kwargs["init_spline_identity"] = False
        conditioner_kwargs["init_zeros"] = True
    elif cfg.flow.couplings_transform_type == "spline":
        conditioner_kwargs["num_bins"] = cfg.flow.spline_num_bins

    shift_gen = torch.Generator().manual_seed(123)
    mask_gen = np.random.default_rng(seed=321)

    if (
        not isinstance(system, MolecularSystem) and shape_info[bg.TARGET][0] == 2
    ):  # This makes expressing the architecture for the GMM system much easier, as we can use "X1" and "X2" to implement swaps
        X1 = bg.TensorInfo("X1", False, True)
        X2 = bg.TensorInfo("X2", False, True)
        builder.add_split(bg.TARGET, (X1, X2), ([0], [1]))

    for item in cfg.flow.architecture:
        if len(item) == 4:
            what_str, on_str, add_reverse, temp_aware = item
        else:
            what_str, on_str, add_reverse = item
            temp_aware = False

        what, on = parse_architecture_layer_str(what_str, on_str, builder.current_dims)
        builder.add_condition(
            what,
            on=on,
            add_reverse=add_reverse,
            rng=mask_gen,
            conditioner_type="dense",
            transformer_type=transformer_type,
            transformer_kwargs=transformer_kwargs,
            context_dims=1 if temp_aware else 0,
            conditioner_factory=conditioner_factory,
            **conditioner_kwargs,
        )
        if cfg.flow.torsion_shifts:
            assert isinstance(
                system, MolecularSystem
            ), "Torsion shifts are only supported for molecular systems"
            builder.add_torsion_shifts(torch.rand((), generator=shift_gen))

    if (
        not isinstance(system, MolecularSystem) and shape_info[bg.TARGET][0] == 2
    ):  # Merge back
        builder.add_merge((X1, X2), bg.TARGET, sizes_or_indices=([0], [1]))

    ##### Build context preprocessor (used for temperature conditioning) #####
    if (
        isinstance(cfg.training.training_mode, ReweightingTrainingModeConfig)
        and cfg.training.training_mode.context_preprocessor is not None
    ):
        scale_to_max_range = (
            cfg.training.training_mode.context_preprocessor.scale_to_max_range
        )
        apply_log = cfg.training.training_mode.context_preprocessor.apply_log
        min_temp = cfg.training.training_mode.max_temperature_range[0]
        max_temp = cfg.training.training_mode.max_temperature_range[1]

        if apply_log:
            min_temp = np.log(min_temp)
            max_temp = np.log(max_temp)

        def context_preprocessor(context):
            if context is not None:
                if apply_log:
                    if torch.is_tensor(context):
                        context = torch.log(context)
                    else:
                        context = np.log(context)
                if scale_to_max_range:
                    context = (context - min_temp) / (max_temp - min_temp)
            return context

    else:
        context_preprocessor = None

    ##### Add chirality constraint layers #####
    if cfg.system.constrain_chirality:
        assert isinstance(
            system, MolecularSystem
        ), "Chirality constraints are only supported for molecular systems"

        chiral_torsions = bgmol.is_chiral_torsion(
            system.coordinate_trafo_openmm.torsion_indices,
            system.openmm_system.mdtraj_topology,
        )
        builder.add_constrain_chirality(chiral_torsions)
        system._NO_constraint_layers += 1

        chiral_torsions_indices = np.argwhere(chiral_torsions).flatten()
        print("Added constraint for chirality of torsions:", chiral_torsions_indices)

    ##### Potentially add layers for constraining hydrogen permutation #####
    if cfg.flow.min_energy_structure_path is None and isinstance(
        system, MolecularSystem
    ):
        print(
            "No path to minimum energy structure given, performing energy minimization..."
        )
        sim = system.openmm_system.create_openmm_simulation(
            platform=Platform.getPlatformByName("CUDA"),
            temperature=cfg.main_temp,
        )
        sim.minimizeEnergy()
        state = sim.context.getState(getPositions=True)
        position = state.getPositions(True).value_in_unit(nanometer)
        min_energy_structure_path = os.path.join(
            "./input_files/", cfg.system.name, "position_min_energy.pt"
        )
        torch.save(
            torch.tensor(
                position.reshape(
                    1, 3 * system.openmm_system.system.getNumParticles()
                ).astype(np.float64)
            ),
            min_energy_structure_path,
        )
        print("New minimum energy structure saved to", min_energy_structure_path)

        del sim
    else:
        min_energy_structure_path = cfg.flow.min_energy_structure_path

    if isinstance(system, MolecularSystem):
        min_energy_structure = torch.load(min_energy_structure_path).to(
            dtype=torch.get_default_dtype()
        )

    if cfg.flow.constrain_hydrogen_permutation:
        assert isinstance(
            system, MolecularSystem
        ), "Hydrogen permutation constraints are only supported for molecular systems"

        constrain_right, constrain_left = (
            get_torsion_indices_for_permutation_constraints(
                system=system,
                reference_minimum_energy_structure_cart=min_energy_structure,
            )
        )

        # Let's "abuse" "add_constrain_chirality", because it does exactly what we want:
        builder.add_constrain_chirality(constrain_right, right_handed=False)
        builder.add_constrain_chirality(constrain_left, right_handed=True)
        system._NO_constraint_layers += 2

    ##### Add IC scaling layer #####

    flow_stop_index = len(builder.layers)

    IC_start_index = len(builder.layers)
    if isinstance(system, MolecularSystem):
        constraints = bond_constraints(
            system.openmm_system.system, system.coordinate_trafo_openmm
        )

        ic_scaler = ICScaler(
            min_energy_structure,
            ic_trafo=system.coordinate_trafo_openmm,
            constrained_bond_indices=constraints[0],
        )
        ic_scaler = ic_scaler.to("cuda")
        builder.add_layer(ic_scaler)

    ##### Add hydrogen bond length constraints #####

    if isinstance(system, MolecularSystem):
        if len(constraints[0]) > 0:
            builder.add_merge_constraints(*constraints)

    ##########

    if isinstance(system, MolecularSystem):
        builder.add_map_to_cartesian(
            coordinate_transform=system.coordinate_trafo_openmm
        )

    IC_stop_index = len(builder.layers)

    generator = builder.build_generator(
        use_sobol=cfg.flow.use_sobol_prior,
        context_preprocessor=context_preprocessor,
    )

    ##### Build generator without the "IC -> Cartesian" transformation #####

    # Note: In case of non-molecular systems, generator_IC will be the same as generator, which is what we want
    layers_without_IC_trafo = generator.flow._blocks[:flow_stop_index]
    generator_IC = BoltzmannGenerator(
        prior=generator._prior,
        flow=SequentialFlow(
            layers_without_IC_trafo,
            context_preprocessor=context_preprocessor,
        ),
        target=generator._target,
    )

    ##### Build flow that only contains the "IC -> Cartesian" transformation #####

    # Note: In case of non-molecular systems, IC_trafo will be an identity transformation, which is what we want
    layers_IC_trafo = generator.flow._blocks[IC_start_index:IC_stop_index]
    IC_trafo = SequentialFlow(
        layers_IC_trafo
    )  # Converts from scaled IC ([0,1]) to cartesian

    ##########

    print(
        "Total number of parameters in the generator",
        sum(p.numel() for p in generator.parameters()),
    )

    return generator, generator_IC, IC_trafo, system
