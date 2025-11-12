import os
import traceback
from copy import deepcopy

import hydra
import torch
import wandb
from auto_slurm.helpers import start_run, write_resume_file
from omegaconf import DictConfig

from annealed_bg.evaluation.core import run_evaluation
from annealed_bg.generator.create_generator import create_generator
from annealed_bg.training_modes.factory import create_training_mode
from annealed_bg.utils.build_config import build_config
from annealed_bg.utils.checkpointing import load_checkpoint, save_checkpoint
from annealed_bg.utils.dataloading import (
    FastTensorDataLoader,
    convert_to_IC,
    load_trajectory,
)
from annealed_bg.utils.general import build_resume_python_command, set_precision
from annealed_bg.utils.matplotlib_utils import setup_matplotlib_defaults
from annealed_bg.utils.newline_tqdm import NewlineTqdm as tqdm
from annealed_bg.utils.permutations import apply_permutation
from annealed_bg.utils.wandb import initialize_wandb


@hydra.main(version_base=None, config_path="configs", config_name="reverse_kl.yaml")
def run(cfg_dict: DictConfig) -> None:

    setup_matplotlib_defaults()

    cfg, cfg_dict = build_config(cfg_dict=cfg_dict)

    set_precision(cfg.general.precision)

    if cfg.experiment.resume_after_in_h is not None:
        experiment_timer = start_run(
            time_limit=cfg.experiment.resume_after_in_h,
        )
    else:
        experiment_timer = None

    initialize_wandb(
        cfg.experiment,
        cfg_dict=(
            cfg_dict
            if not (
                cfg.experiment.only_run_eval
                and cfg.experiment.resume_wandb_id is not None
            )
            else None
        ),
    )

    try:
        ##### Create model (generator) #####

        generator, generator_IC, IC_trafo, system = create_generator(
            cfg=cfg,
        )

        checkpoint_path = (
            cfg.training.checkpoint_path
            if cfg.training.checkpoint_path != "default"
            else cfg.system.default_checkpoint_paths[str(float(cfg.main_temp))]
        )
        if checkpoint_path is not None:
            generator.load_state_dict(torch.load(checkpoint_path)["model"])

        ##### Create optimizer and lr schedulers #####

        optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=cfg.training.lr,
            weight_decay=(
                cfg.training.weight_decay
                if cfg.training.weight_decay is not None
                else 0
            ),
        )

        if cfg.training.lr_scheduler is not None:
            lr_scheduler = cfg.training.lr_scheduler.create_scheduler(optimizer)
        else:
            lr_scheduler = None

        if cfg.training.warmup_iters is not None:
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda s: min(1.0, s / cfg.training.warmup_iters)
            )
        else:
            warmup_scheduler = None

        ##### Store initial generator and optimizer state #####

        # This is needed later, since the reweighting training mode with sequence temp-sampling allows resetting the
        # generator (and optimizer) to its initial state at the beginning of a sequence step.
        generator_initial_state = {
            k: v.cpu() for k, v in generator.state_dict().items()
        }
        generator_initial_state = deepcopy(generator_initial_state)
        optimizer_initial_state = {k: v for k, v in optimizer.state_dict().items()}
        optimizer_initial_state = deepcopy(optimizer_initial_state)

        ##### Create training mode object #####

        def reinit_fn():
            generator.load_state_dict(generator_initial_state)
            optimizer.load_state_dict(optimizer_initial_state)

        training_mode = create_training_mode(
            training_mode_config=cfg.training.training_mode,
            generator=generator,
            generator_IC=generator_IC,
            IC_trafo=IC_trafo,
            system=system,
            system_cfg=cfg.system,
            batch_size=cfg.training.batch_size,
            reinit_fn=reinit_fn,
            optimizer=optimizer,
            main_temp=cfg.main_temp,
        )

        ##### Convert validation datasets to IC space #####

        eval_sampling_T_pairs = (
            training_mode.eval_sampling_T_pairs
            if cfg.evaluation.overwrite_eval_sampling_T_pairs is None
            else cfg.evaluation.overwrite_eval_sampling_T_pairs
        )

        val_data_needed = [cfg.main_temp]

        for item in eval_sampling_T_pairs:
            if item[1] is not None:
                val_data_needed.append(item[1])
            elif item[0] is not None:
                val_data_needed.append(item[0])
            else:
                # If they are both None, we compare with main_temp:
                val_data_needed.append(cfg.main_temp)
        val_data_needed = list(set(val_data_needed))

        val_datasets_cart = {}
        val_datasets_IC = {}
        for T in val_data_needed:
            path = cfg.system.val_data[str(float(T))]
            val_datasets_cart[T] = load_trajectory(path, data_type=cfg.system.data_type)

        if not cfg.evaluation.apply_cart_permutation_to_ground_truth_datasets:
            for T in val_datasets_cart.keys():
                val_datasets_IC[T] = convert_to_IC(
                    val_datasets_cart[T], IC_trafo=IC_trafo
                )
            # Otherwise, we first apply the permutation to the cartesian datasets and only then
            # convert to IC space - but we do this later.

        ##### If necessary, load checkpoint #####
        if cfg.experiment.resume_wandb_id is not None:
            starting_i, checkpoint_path = load_checkpoint(
                cfg=cfg.experiment,
                generator=generator,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                warmup_scheduler=warmup_scheduler,
            )
            if not cfg.experiment.only_run_eval:  # Otherwise we don't need this
                training_mode.load(dir_path=os.path.dirname(checkpoint_path))
        else:
            starting_i = 0
            checkpoint_path = None

        ##### If in "only_run_eval" mode, run evaluation and exit #####

        if cfg.experiment.only_run_eval:
            if cfg.experiment.eval_outdir is None:
                print("Writing eval to wandb step", wandb.run.step)

            run_evaluation(
                cfg=cfg,
                training_mode=training_mode,
                generator=generator,
                IC_trafo=IC_trafo,
                val_datasets_IC=val_datasets_IC,
                val_datasets_cart=val_datasets_cart,
                apply_cart_permutation_to_ground_truth_datasets=cfg.evaluation.apply_cart_permutation_to_ground_truth_datasets,
                eval_sampling_T_pairs=eval_sampling_T_pairs,
                n_samples=cfg.evaluation.big_eval_samples,  # Do a "big" evaluation
                system=system,
                eval_outdir=cfg.experiment.eval_outdir,
                current_i=(
                    wandb.run.step if cfg.experiment.eval_outdir is None else None
                ),
                do_evaluate_sampling=cfg.experiment.only_run_eval_include_sampling,
                do_evaluate_additional_sampling=cfg.experiment.only_run_eval_include_additional_sampling,
                do_evaluate_NLL=cfg.experiment.only_run_eval_include_NLL,
                do_evaluate_forward_ESS=cfg.experiment.only_run_eval_include_forward_ESS,
                is_initial_eval=False,
                is_big_eval=True,
            )

            try:
                if generator._target._bridge is not None:
                    generator._target._bridge.context_wrapper.terminate()
            except:
                pass

            return

        ##### Run initial evaluation to show the initial flow distribution #####
        if (
            cfg.experiment.resume_wandb_id is None
            and not cfg.evaluation.skip_initial_eval
        ):  # Only do this if we are not resuming
            run_evaluation(
                cfg=cfg,
                training_mode=training_mode,
                generator=generator,
                IC_trafo=IC_trafo,
                val_datasets_IC=val_datasets_IC,
                val_datasets_cart=val_datasets_cart,
                apply_cart_permutation_to_ground_truth_datasets=cfg.evaluation.apply_cart_permutation_to_ground_truth_datasets,
                eval_sampling_T_pairs=[
                    item for item in eval_sampling_T_pairs if item[1] is None
                ],  # Skip the evals where we would be reweighting
                n_samples=cfg.evaluation.eval_samples,  # For this initial eval, do a "small" evaluation
                system=system,
                eval_outdir=None,  # Log to wandb, not to disk
                current_i=0,
                do_evaluate_sampling=True,
                do_evaluate_additional_sampling=True,
                do_evaluate_NLL=False,  # Does not make much sense for a randomly initialized model
                do_evaluate_forward_ESS=False,  # Does not make much sense for a randomly initialized model
                is_initial_eval=True,
                is_big_eval=False,
            )

        ##### Load training data if needed #####
        if training_mode.needs_samples:
            train_data = load_trajectory(
                cfg.system.train_data[str(float(cfg.main_temp))],
                data_type=cfg.system.data_type,
            )
            train_loader = FastTensorDataLoader(
                train_data,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                drop_last=True,
            )
            train_iter = iter(train_loader)

        ##########

        summed_metrics = {}
        counter = 0

        for i in tqdm(
            range(
                starting_i,
                cfg.training.max_iter,
            ),
            mininterval=30.0,
        ):
            current_metrics = {}

            optimizer.zero_grad()

            if training_mode.needs_samples:
                try:
                    batch = next(train_iter)[0]
                except StopIteration:
                    train_iter = iter(train_loader)  # Also shuffles the data
                    batch = next(train_iter)[0]
                batch = batch.to("cuda")
            else:
                batch = None

            ##### Calculate loss #####

            try:

                loss, additional_metrics = training_mode.calculate_loss(
                    current_i=i, batch=batch
                )

            except Exception as e:
                print("Exception caught in loss calculation in step", i)
                # Print the full stacktrace:
                print(traceback.format_exc())
                print("Skipping iteration.")
                continue

            ####################

            current_metrics["train_loss"] = loss.item()
            current_metrics.update(additional_metrics)

            loss.backward()

            ##### Gradient clipping #####
            if cfg.training.max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    generator.parameters(), cfg.training.max_grad_norm
                )
                current_metrics["grad_norm"] = grad_norm.item()
            ####################

            # Only apply the update if there are no NaNs or INFs in the gradients / loss:
            if (
                not all(
                    torch.all(torch.isfinite(p.grad))
                    for p in generator.parameters()
                    if p.grad is not None
                )
            ) or (not torch.all(torch.isfinite(loss))):
                print(
                    "Found NaN or INF in gradient or loss, skipping optimization step."
                )
            else:
                optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            if warmup_scheduler is not None and i <= cfg.training.warmup_iters:
                warmup_scheduler.step()

            for key, value in current_metrics.items():
                if key not in summed_metrics:
                    summed_metrics[key] = 0
                summed_metrics[key] += value
            counter += 1

            ##### Wandb log step #####
            if (i + 1) % cfg.general.wandb_log_every == 0:
                wandb.log(
                    {
                        "lr": optimizer.param_groups[0]["lr"],
                        "energy_calls_counter": training_mode.energy_calls_counter,
                    },
                    step=i,
                )

                if training_mode.state_to_log != {}:
                    wandb.log(
                        training_mode.state_to_log,
                        step=i,
                    )

                for key, value in summed_metrics.items():
                    summed_metrics[key] = value / counter
                wandb.log(summed_metrics, step=i)

                summed_metrics = {}
                counter = 0

            ##### Write checkpoint #####

            time_limit_reached = (
                experiment_timer is not None and experiment_timer.time_limit_reached()
            )

            if (
                (i + 1) % cfg.checkpointing.write_checkpoint_every == 0
                or ((i + 1) == cfg.training.max_iter)
                or time_limit_reached
            ):
                save_dir = save_checkpoint(
                    current_i=i,
                    generator=generator,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    warmup_scheduler=warmup_scheduler,
                )
                training_mode.save(dir_path=save_dir)

            ##### Run evaluation #####

            if cfg.evaluation.apply_cart_permutation_to_ground_truth_datasets:
                # Determine the best permutation of hydrogens in CH_X groups and apply it to the validation datasets:

                if (i == starting_i) or (
                    (not training_mode.only_determine_permutation_once)
                    and (
                        (i + 1) % cfg.evaluation.eval_every == 0
                        or (i + 1) % cfg.evaluation.big_eval_every == 0
                        or (i + 1) % cfg.evaluation.NLL_every == 0
                    )
                ):
                    apply_permutation(
                        val_datasets_IC=val_datasets_IC,
                        val_datasets_cart=val_datasets_cart,
                        generator=generator,
                        IC_trafo=IC_trafo,
                        system=system,
                        main_temp=cfg.main_temp,
                    )

            if (
                (i + 1) % cfg.evaluation.eval_every == 0
                or (i + 1) % cfg.evaluation.big_eval_every == 0
                or (i + 1) % cfg.evaluation.NLL_every == 0
                or (i + 1) % cfg.evaluation.additional_sampling_eval_every == 0
            ):
                try:
                    run_evaluation(
                        cfg=cfg,
                        training_mode=training_mode,
                        generator=generator,
                        IC_trafo=IC_trafo,
                        val_datasets_IC=val_datasets_IC,
                        val_datasets_cart=val_datasets_cart,
                        apply_cart_permutation_to_ground_truth_datasets=False,  # Already applied above, so we can skip it here
                        eval_sampling_T_pairs=eval_sampling_T_pairs,
                        n_samples=(
                            cfg.evaluation.big_eval_samples
                            if (i + 1) % cfg.evaluation.big_eval_every == 0
                            else cfg.evaluation.eval_samples
                        ),
                        system=system,
                        eval_outdir=None,  # Log to wandb
                        current_i=i,
                        do_evaluate_sampling=(i + 1) % cfg.evaluation.eval_every == 0
                        or (i + 1) % cfg.evaluation.big_eval_every == 0,
                        do_evaluate_additional_sampling=(i + 1)
                        % cfg.evaluation.additional_sampling_eval_every
                        == 0
                        or (i + 1) % cfg.evaluation.big_eval_every == 0,
                        do_evaluate_NLL=(i + 1) % cfg.evaluation.NLL_every == 0,
                        do_evaluate_forward_ESS=(i + 1) % cfg.evaluation.big_eval_every
                        == 0,  # Only calculate forward ESS for big evals
                        is_initial_eval=False,
                        is_big_eval=((i + 1) % cfg.evaluation.big_eval_every == 0),
                    )
                except Exception as e:
                    print("Exception caught in evaluation in step", i)
                    # Print the full stacktrace:
                    print(traceback.format_exc())
                    print("Skipping evaluation step.")

            if time_limit_reached:
                print(
                    f"Reached {cfg.experiment.resume_after_in_h} hours. Stopping training, to be resumed in the next job."
                )
                break

        ##### Write resume file to pick up remaining work #####

        if time_limit_reached and i < cfg.training.max_iter - 1:  # Work not done
            command = build_resume_python_command()
            write_resume_file(command)

            print(
                "Wrote resume file to be picked up by next job at iteration",
                i + 1,
            )

        ####################

    finally:
        try:
            wandb.finish()
            if generator._target._bridge is not None:
                generator._target._bridge.context_wrapper.terminate()
        except:
            pass


if __name__ == "__main__":
    run()
