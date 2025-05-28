from typing import Literal

import bgflow as bg
import bgmol
import torch
from bgflow import ANGLES, BONDS

from annealed_bg.config.training import EnergyRegularizerConfig
from annealed_bg.evaluation.plot import plot_IC_marginals
from annealed_bg.systems.base import System
from annealed_bg.systems.molecular.utils import get_openmm_system
from annealed_bg.utils.energy_regularizers import lin_log_cut


class MolecularSystem(System):
    def __init__(
        self,
        system_name: str,
        energy_regularizer_cfg: EnergyRegularizerConfig | None,
        n_workers: int,
        system_temp: float,
    ):
        self.system_name = system_name
        self.energy_regularizer_cfg = energy_regularizer_cfg
        self.system_temp = system_temp

        self.openmm_system = get_openmm_system(system_name, n_workers, system_temp)

        self._create_coordinate_trafo()
        self._initialize_base()

        self._NO_constraint_layers = 0

    @property
    def NO_constraint_layers(self) -> int:
        return self._NO_constraint_layers

    def _create_coordinate_trafo(self):
        if self.system_name == "aldp":
            z_matrix = bgmol.systems.ala2.DEFAULT_GLOBAL_Z_MATRIX
        else:
            z_matrix = bgmol.ZMatrixFactory(
                self.openmm_system.mdtraj_topology
            ).build_with_templates()[0]

        self.coordinate_trafo_openmm = bg.GlobalInternalCoordinateTransformation(
            z_matrix=z_matrix,
            enforce_boundaries=True,
            normalize_angles=True,
        )

        print(
            "Number of bond constraints:", self.openmm_system.system.getNumConstraints()
        )
        self._shape_info = bg.ShapeDictionary.from_coordinate_transform(
            self.coordinate_trafo_openmm,
            n_constraints=self.openmm_system.system.getNumConstraints(),  # Pass number of hydrogen bond length constraints
        )

    def _initialize_base(self):
        energy_high = (
            self.energy_regularizer_cfg.energy_high
            if self.energy_regularizer_cfg is not None
            else None
        )
        energy_max = (
            self.energy_regularizer_cfg.energy_max
            if self.energy_regularizer_cfg is not None
            else None
        )

        # Make sure that either both energy_high and energy_max are None or both are set
        assert (energy_high is None) == (
            energy_max is None
        ), "Either both energy_high and energy_max must be None or both must be set."
        print(
            "Setting up energy regularizer with high energy",
            energy_high,
            "and max energy",
            energy_max,
        )

        energy_fn = self.openmm_system.energy_model.energy

        if energy_high is not None:
            energy_regularizer_fn = lambda energy: lin_log_cut(
                energy, energy_high, energy_max
            )
        else:
            energy_regularizer_fn = lambda energy: energy

        def wrapped_energy_fn(xs, temperature=None):
            if temperature is None:
                return energy_regularizer_fn(energy_fn(xs))
            else:
                return (
                    energy_regularizer_fn(energy_fn(xs))
                    * self.system_temp
                    / temperature
                )

        super().__init__(
            wrapped_energy_fn,
            torch.Size([self.openmm_system.system.getNumParticles() * 3]),
            self.openmm_system.energy_model._bridge,
        )

    @property
    def IC_shape_info(self) -> bg.ShapeDictionary:
        return self._shape_info

    @property
    def spline_range(self) -> tuple[float, float] | None:
        return [0.0, 1.0]

    def get_prior_type_and_kwargs(
        self, transform_type: Literal["spline", "rnvp"]
    ) -> tuple[dict, dict]:

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

        # Note: We used the bgflow default (uniform distribution [0,1]) for torsions

        return prior_type, prior_kwargs

    def plot_IC_marginals(
        self,
        flow_data: torch.Tensor | None,
        ground_truth_data: torch.Tensor,
        tag: str,
        current_i: int | None = None,
        flow_data_weights: torch.Tensor | None = None,
        plot_as_free_energy: bool = False,
        marginals_2D: list | None = None,
        marginals_2D_vmax: float = 6.5,
        dpi_marginals=50,
        dpi_2D_marginal=100,
        report_wandb: bool = False,
        output_dir: str | None = None,
        do_calculate_2D_marginal_kld: bool = False,
        is_big_eval: bool = False,
    ):
        plot_IC_marginals(
            flow_data,
            ground_truth_data,
            tag,
            current_i,
            flow_data_weights,
            [0.0, 1.0],
            plot_as_free_energy,
            marginals_2D,
            marginals_2D_vmax,
            dpi_marginals,
            dpi_2D_marginal,
            report_wandb,
            output_dir,
            do_plot_1D_marginals=is_big_eval,
            do_calculate_2D_marginal_kld=do_calculate_2D_marginal_kld,
        )
