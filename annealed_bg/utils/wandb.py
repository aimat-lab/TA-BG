import io
import lzma
import os
import pickle
from glob import glob
from typing import Tuple

import matplotlib.pyplot as plt
import wandb
from matplotlib.figure import Figure
from PIL import Image

from annealed_bg.config.experiment import ExperimentConfig


def initialize_wandb(cfg: ExperimentConfig, cfg_dict: dict = None):
    """Initialize wandb experiment and print run information.

    Args:
        cfg: The experiment configuration.
        cfg_dict: The configuration dictionary to log to wandb.
    """

    notes = (
        cfg.wandb_notes
        if not (cfg.only_run_eval and cfg.resume_wandb_id is not None)
        else None
    )
    if notes is not None:
        if cfg.resume_wandb_id is not None and cfg.wandb_force_new_experiment:
            notes = "Resumed from " + cfg.resume_wandb_id + "; " + notes

    wandb.init(
        project="annealed_bg",
        notes=notes,
        group=cfg.wandb_group,
        tags=cfg.wandb_tags if not cfg.only_run_eval else None,
        mode=(
            "online"
            if not (
                cfg.only_run_eval and cfg.eval_outdir is not None
            )  # In this case we only run evaluation and save the results to disk
            and not (cfg.disable_wandb or cfg.wandb_offline)
            else ("offline" if cfg.wandb_offline else "disabled")
        ),
        settings=wandb.Settings(
            console_multipart=True,  # needed for resume jobs not to overwrite previous logs
        ),
        resume=(
            "must"
            if (
                cfg.resume_wandb_id is not None
                and not (
                    cfg.only_run_eval and cfg.eval_outdir is not None
                )  # In this case we only run evaluation and save the results to disk
                and not cfg.wandb_force_new_experiment
            )
            else None
        ),
        id=(
            cfg.resume_wandb_id
            if not (
                cfg.only_run_eval and cfg.eval_outdir is not None
            )  # In this case we only run evaluation and save the results to disk
            and not cfg.wandb_force_new_experiment
            else None
        ),
        allow_val_change=True,
        config=cfg_dict,
    )

    info_str = "Run information:\n"
    if cfg.wandb_notes is not None:
        info_str += "Notes: " + cfg.wandb_notes + "\n"
    info_str += "Wandb id: " + wandb.run.id + "\n"
    info_str += "Wandb run dir: " + wandb.run.dir + "\n"
    info_str += (
        "Wandb run name: "
        + (wandb.run.name if wandb.run.name is not None else "None")
        + "\n"
    )
    info_str += "Slurm job id: " + os.environ.get("SLURM_JOB_ID", "None") + "\n"
    info_str += (
        "Task index: " + os.environ.get("SLURM_SUBMIT_TASK_INDEX", "None") + "\n"
    )
    print(info_str, flush=True)

    if not (cfg.only_run_eval and cfg.eval_outdir is not None):
        os.makedirs(f"{wandb.run.dir}/figures_pickled", exist_ok=True)


def log_figure(
    tag: str,
    current_i: int,
    fig: Figure | None = None,
    write_pickle: bool = True,
    report_wandb: bool = True,
    output_dir: str | None = None,
    dpi: int = 300,
):
    """
    Log a figure.
    - If report_wandb is True, log the figure to wandb.
    - If output_dir is not None, save the figure to the output_dir.
    - If write_pickle is True, pickle the figure to the disk for later use / replotting.

    Args:
        tag: Description of the figure.
        current_i: The current iteration.
        fig: The figure to log. If None, we use plt.gcf().
        write_pickle: Whether to pickle the figure and save to wandb dir.
        report_wandb: Whether to log the figure to wandb.
        output_dir: The output directory.
        dpi: The dpi to use.
    """

    if fig is None:
        fig = plt.gcf()

    if report_wandb:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)

        wandb.log(
            {
                tag: wandb.Image(Image.open(buf)),
            },
            step=current_i,
        )

        if write_pickle:
            with lzma.open(
                f"{wandb.run.dir}/figures_pickled/{tag}_{current_i}.pkl", "wb"
            ) as f:
                pickle.dump(fig, f)

    if output_dir is not None:
        fig.savefig(
            os.path.join(output_dir, f"{tag}_{current_i}.png"),
            dpi=dpi,
            bbox_inches="tight",
        )

        if write_pickle:
            with lzma.open(
                os.path.join(output_dir, f"{tag}_{current_i}.pkl"), "wb"
            ) as f:
                pickle.dump(fig, f)


def get_newest_checkpoint_from_wandb_id(
    wandb_id: str, checkpoint_i: int = None
) -> Tuple[str, int]:
    """Given a wandb_id, returns the path to the newest checkpoint.
    If checkpoint_i is provided, returns the path to that checkpoint.

    Args:
        wandb_id: The wandb id.
        checkpoint_i: The checkpoint iteration.

    Returns:
        The path to the checkpoint and the iteration of the checkpoint.
    """

    wandb_main_dir = os.path.join(os.environ.get("WANDB_DIR", "."), "wandb")
    wandb_run_dirs = glob(os.path.join(wandb_main_dir, f"*run-*-{wandb_id}"))

    assert (
        len(wandb_run_dirs) > 0
    ), f"Resuming failed: No wandb run directories with id {wandb_id} found."

    all_checkpoint_paths = []
    all_checkpoint_iterations = []
    for wandb_run_dir in wandb_run_dirs:
        checkpoints_dir = os.path.join(wandb_run_dir, "files", "checkpoints")

        if not os.path.exists(checkpoints_dir):
            continue

        checkpoint_files = os.listdir(checkpoints_dir)
        checkpoint_files, checkpoint_iterations = zip(
            *[
                (file, int(file.split("_")[1].split(".")[0]))
                for i, file in enumerate(checkpoint_files)
                if ("checkpoint" in file and file.endswith(".pt"))
            ]
        )
        all_checkpoint_paths.extend(
            [os.path.join(checkpoints_dir, file) for file in checkpoint_files]
        )
        all_checkpoint_iterations.extend(checkpoint_iterations)

    assert len(all_checkpoint_paths) > 0, "Resuming failed: No checkpoints found."

    if checkpoint_i is not None:
        assert (
            checkpoint_i in all_checkpoint_iterations
        ), f"Resuming failed: Checkpoint {checkpoint_i} not found."
        checkpoint_path = all_checkpoint_paths[
            all_checkpoint_iterations.index(checkpoint_i)
        ]

        if checkpoint_i != max(all_checkpoint_iterations):
            print(
                f"Warning: The specified checkpoint_i ({checkpoint_i}) is not the latest checkpoint ({max(all_checkpoint_iterations)}).",
                "This can cause problems with wandb logging.",
            )
    else:  # Just take the checkpoint with the largest iteration
        checkpoint_i = max(all_checkpoint_iterations)
        checkpoint_path = all_checkpoint_paths[
            all_checkpoint_iterations.index(checkpoint_i)
        ]

    return checkpoint_path, checkpoint_i
