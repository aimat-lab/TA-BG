import time
import traceback
from typing import Tuple

import wandb
from bgflow import BoltzmannGenerator
from bgflow.nn.flow.sequential import SequentialFlow

from annealed_bg.config.main import Config
from annealed_bg.evaluation.sampling import evaluate_sampling
from annealed_bg.evaluation.utils import (
    evaluate_forward_ESS_batched,
    evaluate_metric_batched,
)
from annealed_bg.systems.base import System
from annealed_bg.training_modes.base import TrainingMode
from annealed_bg.utils.permutations import apply_permutation


def run_evaluation(
    cfg: Config,
    training_mode: TrainingMode,
    generator: BoltzmannGenerator,
    IC_trafo: SequentialFlow,
    val_datasets_IC: dict,
    val_datasets_cart: dict,
    apply_cart_permutation_to_ground_truth_datasets: bool,
    eval_sampling_T_pairs: list[Tuple[float | None, float | None]],
    n_samples: int,
    system: System,
    eval_outdir: str | None = None,
    current_i: int | None = None,
    do_evaluate_sampling: bool = True,
    do_evaluate_additional_sampling: bool = True,
    do_evaluate_NLL: bool = True,
    do_evaluate_forward_ESS: bool = True,
    is_initial_eval: bool = False,
    is_big_eval: bool = False,
):
    if apply_cart_permutation_to_ground_truth_datasets:
        apply_permutation(
            val_datasets_IC=val_datasets_IC,
            val_datasets_cart=val_datasets_cart,
            generator=generator,
            IC_trafo=IC_trafo,
            system=system,
            main_temp=cfg.main_temp,
        )

    if do_evaluate_sampling:
        evaluate_sampling(
            generator=generator,
            IC_trafo=IC_trafo,
            n_samples=n_samples,
            val_datasets_IC=val_datasets_IC,
            val_datasets_cart=val_datasets_cart,
            eval_sampling_T_pairs=eval_sampling_T_pairs,
            main_temp=cfg.main_temp,
            current_i=current_i,
            system=system,
            system_cfg=cfg.system,
            report_wandb=eval_outdir is None,
            output_dir=eval_outdir,
            do_calculate_2D_marginal_kld=True,
            do_calculate_ESS=True,
            is_big_eval=is_big_eval,
        )

    if do_evaluate_additional_sampling:
        training_mode.run_additional_sampling_evaluation(
            is_initial_eval=is_initial_eval,
            eval_config=cfg.evaluation,
            system=system,
            system_cfg=cfg.system,
            val_datasets_IC=val_datasets_IC,
            current_i=current_i,
            marginals_2D=cfg.system.marginals_2D,
            marginals_2D_vmax=cfg.system.marginals_2D_vmax,
            report_wandb=eval_outdir is None,
            output_dir=eval_outdir,
        )

    if do_evaluate_NLL:
        start = time.time()

        eval_NLL_Ts = (
            training_mode.eval_NLL_Ts
            if cfg.evaluation.overwrite_eval_NLL_Ts is None
            else cfg.evaluation.overwrite_eval_NLL_Ts
        )

        for T in eval_NLL_Ts:
            print(
                "Calculating NLL for T =",
                T,
                "with",
                n_samples,
                "samples...",
            )

            try:
                NLL = evaluate_metric_batched(
                    generator.energy,
                    val_datasets_cart[T if T is not None else cfg.main_temp][
                        :n_samples
                    ],
                    context_temp=T,
                )
            except Exception as e:
                print(
                    "Exception caught when calculating NLL for T=",
                    T,
                    "in step",
                    current_i,
                )
                # Print the full stacktrace:
                print(traceback.format_exc())
                print("Skipping NLL calculation.")
                break

            if eval_outdir is not None:
                print(f"NLL_{float(T) if T is not None else 'None'}:", NLL)
            else:
                wandb.log(
                    {f"NLL_{float(T) if T is not None else 'None'}": NLL}, current_i
                )

        print("NLL calculation took", time.time() - start, "seconds.")

    if do_evaluate_forward_ESS:
        # Evaluate forward ESS:
        if len(cfg.evaluation.calculate_forward_ESS_for_Ts) > 0:
            start = time.time()
            for T in cfg.evaluation.calculate_forward_ESS_for_Ts:
                print("Calculating forward ESS for T =", T)

                try:
                    all_ESS = evaluate_forward_ESS_batched(
                        generator=generator,
                        temp=T,
                        val_data_cart=val_datasets_cart[T][:n_samples],
                    )
                except Exception as e:
                    print(
                        "Exception caught when calculating forward ESS for T=",
                        T,
                        "in step",
                        current_i,
                    )
                    # Print the full stacktrace:
                    print(traceback.format_exc())
                    print("Skipping forward ESS calculation.")
                    break

                for clipping_fraction, ESS in all_ESS.items():
                    clipping_str = (
                        f"_{clipping_fraction}" if clipping_fraction is not None else ""
                    )
                    if eval_outdir is not None:
                        print(f"FWD_ESS_{float(T)}{clipping_str}:", ESS)
                    else:
                        wandb.log(
                            {f"FWD_ESS_{float(T)}{clipping_str}": ESS},
                            current_i,
                        )

            print("Forward ESS calculation took", time.time() - start, "seconds.")
