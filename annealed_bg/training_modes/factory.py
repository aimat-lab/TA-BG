import torch
from bgflow import BoltzmannGenerator
from bgflow.nn.flow.sequential import SequentialFlow

from annealed_bg.config.system import SystemConfig
from annealed_bg.config.training_modes.base import TrainingModeConfig
from annealed_bg.config.training_modes.fab import FabTrainingModeConfig
from annealed_bg.config.training_modes.forward_kld import ForwardKLDTrainingModeConfig
from annealed_bg.config.training_modes.reverse_kld import ReverseKLDTrainingModeConfig
from annealed_bg.config.training_modes.reweighting import ReweightingTrainingModeConfig
from annealed_bg.systems.base import System
from annealed_bg.training_modes.fab import FABTrainingMode
from annealed_bg.training_modes.forward_kld import ForwardKLDTrainingMode
from annealed_bg.training_modes.reverse_kld import ReverseKLDTrainingMode
from annealed_bg.training_modes.reweighting import ReweightingTrainingMode


def create_training_mode(
    training_mode_config: TrainingModeConfig,
    generator: BoltzmannGenerator,
    generator_IC: BoltzmannGenerator,
    IC_trafo: SequentialFlow,
    system: System,
    system_cfg: SystemConfig,
    batch_size: int,
    reinit_fn: callable,
    optimizer: torch.optim.Optimizer,
    main_temp: float,
):
    if isinstance(training_mode_config, FabTrainingModeConfig):
        return FABTrainingMode(
            config=training_mode_config,
            generator_IC=generator_IC,
            IC_trafo=IC_trafo,
            system=system,
            batch_size=batch_size,
            main_temp=main_temp,
        )
    elif isinstance(training_mode_config, ReweightingTrainingModeConfig):
        return ReweightingTrainingMode(
            config=training_mode_config,
            generator=generator,
            batch_size=batch_size,
            reinit_fn=reinit_fn,
            optimizer=optimizer,
            main_temp=main_temp,
            cart_dims=system.event_shapes[0],
            eval_IS_clipping_fraction=system_cfg.eval_IS_clipping_fraction,
        )
    elif isinstance(training_mode_config, ForwardKLDTrainingModeConfig):
        return ForwardKLDTrainingMode(
            config=training_mode_config, generator=generator, main_temp=main_temp
        )
    elif isinstance(training_mode_config, ReverseKLDTrainingModeConfig):
        return ReverseKLDTrainingMode(
            config=training_mode_config,
            generator=generator,
            batch_size=batch_size,
            main_temp=main_temp,
        )
    else:
        raise ValueError("Unknown training mode configuration.")
