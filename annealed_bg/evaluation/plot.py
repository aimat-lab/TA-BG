import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from annealed_bg.evaluation.metrics import calculate_2D_marginal_kld
from annealed_bg.utils.plot_free_energy import plot_1D_marginal, plot_2D_free_energy
from annealed_bg.utils.wandb import log_figure


def plot_tica(
    ticas_flow: torch.Tensor | None,
    ticas_ground_truth: torch.Tensor,
    flow_data_weights: torch.Tensor | None = None,
    tag: str | None = None,
    current_i: int | None = None,
    dpi=100,
    report_wandb: bool = False,
    output_dir: str | None = None,
    vmax: float | None = None,
):
    assert ticas_flow is None or (
        len(ticas_flow.shape) == 2 and ticas_flow.shape[1] == 2
    )
    assert len(ticas_ground_truth.shape) == 2 and ticas_ground_truth.shape[1] == 2

    counter = 0
    if ticas_flow is not None:
        fig, axs = plt.subplots(1, 2, figsize=(24, 10))

        plot_2D_free_energy(
            ticas_flow[..., 0],
            ticas_flow[..., 1],
            weights=flow_data_weights,
            ax=axs[0],
            cmap=None,  # Use default
            nbins=100,
            vmax=vmax,
            range=[
                [
                    torch.min(ticas_ground_truth[..., 0]).item(),
                    torch.max(ticas_ground_truth[..., 0]).item(),
                ],
                [
                    torch.min(ticas_ground_truth[..., 1]).item(),
                    torch.max(ticas_ground_truth[..., 1]).item(),
                ],
            ],
        )
        axs[0].set_title("Flow")
        axs[0].set_xlabel(f"tica 0")
        axs[0].set_ylabel(f"tica 1")
        axs[0].set_xlim(
            torch.min(ticas_ground_truth[..., 0]).item(),
            torch.max(ticas_ground_truth[..., 0]).item(),
        )
        axs[0].set_ylim(
            torch.min(ticas_ground_truth[..., 1]).item(),
            torch.max(ticas_ground_truth[..., 1]).item(),
        )

        counter += 1
    else:
        fig, axs = plt.subplots(1, 1, figsize=(12, 10))
        axs = [axs]

    plot_2D_free_energy(
        ticas_ground_truth[..., 0],
        ticas_ground_truth[..., 1],
        ax=axs[counter],
        cmap=None,  # Use default
        nbins=100,
        vmax=vmax,
        range=[
            [
                torch.min(ticas_ground_truth[..., 0]).item(),
                torch.max(ticas_ground_truth[..., 0]).item(),
            ],
            [
                torch.min(ticas_ground_truth[..., 1]).item(),
                torch.max(ticas_ground_truth[..., 1]).item(),
            ],
        ],
    )
    axs[counter].set_title("Ground truth")
    axs[counter].set_xlabel(f"tica 0")
    axs[counter].set_ylabel(f"tica 1")
    axs[counter].set_xlim(
        torch.min(ticas_ground_truth[..., 0]).item(),
        torch.max(ticas_ground_truth[..., 0]).item(),
    )
    axs[counter].set_ylim(
        torch.min(ticas_ground_truth[..., 1]).item(),
        torch.max(ticas_ground_truth[..., 1]).item(),
    )

    plt.tight_layout()

    log_figure(
        tag=f"{tag}_tica",
        current_i=current_i,
        fig=plt.gcf(),
        write_pickle=True,
        report_wandb=report_wandb,
        output_dir=output_dir,
        dpi=dpi,
    )


def plot_IC_marginals(
    flow_data: torch.Tensor | None,
    ground_truth_data: torch.Tensor,
    tag: str,
    current_i: int | None = None,
    flow_data_weights: torch.Tensor | None = None,
    data_range: list | None = None,
    plot_as_free_energy: bool = False,
    marginals_2D: list | None = None,
    marginals_2D_vmax: float = 6.5,
    dpi_marginals=50,
    dpi_2D_marginal=100,
    report_wandb: bool = False,
    output_dir: str | None = None,
    do_plot_1D_marginals: bool = True,
    do_calculate_2D_marginal_kld: bool = False,
):
    if do_calculate_2D_marginal_kld:
        assert flow_data is not None

    # Determine data_range using ground truth data if not specified
    if data_range is None:
        data_range = [None, None]

    if data_range[0] is None:
        data_range[0] = torch.min(ground_truth_data).item()
    if data_range[1] is None:
        data_range[1] = torch.max(ground_truth_data).item()

    index_phi_psi_label_mapping = {}
    if marginals_2D is not None:
        for i, marginal_2D in enumerate(marginals_2D):
            index_phi_psi_label_mapping[marginal_2D[0]] = r"$\phi_" + str(i + 1) + "$"
            index_phi_psi_label_mapping[marginal_2D[1]] = r"$\psi_" + str(i + 1) + "$"

    if do_plot_1D_marginals:
        n_bins = 100

        n_dims = ground_truth_data.shape[1]
        n_rows = (n_dims + 6) // 7

        fig, axs = plt.subplots(n_rows, 7, figsize=(5 * 7, 5 * n_rows))
        axs = axs.flatten()

        for i in range(n_dims):
            plot_1D_marginal(
                ground_truth_data[:, i],
                plot_as_free_energy=plot_as_free_energy,
                ax=axs[i],
                n_bins=n_bins,
                label="Ground truth",
                linestyle="-",
            )
            if flow_data is not None:
                plot_1D_marginal(
                    flow_data[:, i],
                    weights=None,
                    plot_as_free_energy=plot_as_free_energy,
                    ax=axs[i],
                    n_bins=n_bins,
                    label="Flow",
                    linestyle="--",
                )
                if flow_data_weights is not None:
                    plot_1D_marginal(
                        flow_data[:, i],
                        weights=flow_data_weights,
                        plot_as_free_energy=plot_as_free_energy,
                        ax=axs[i],
                        n_bins=n_bins,
                        label="Flow (reweighted)",
                        linestyle="dotted",
                    )

            if i not in index_phi_psi_label_mapping:
                xlabel = f"dim {i}"
            else:
                xlabel = index_phi_psi_label_mapping[i] + f" (dim {i})"
            axs[i].set_xlabel(xlabel)

            if plot_as_free_energy:
                axs[i].set_ylabel(r"free energy / $k_\text{B} T$")
            else:
                axs[i].set_ylabel("p")
            axs[i].legend()
            axs[i].set_xlim(data_range)

        # Hide empty subplots
        for i in range(n_dims, n_rows * 7):
            axs[i].axis("off")

        plt.tight_layout()

        log_figure(
            tag=f"{tag}_marginals",
            current_i=current_i,
            fig=fig,
            write_pickle=True,
            report_wandb=report_wandb,
            output_dir=output_dir,
            dpi=dpi_marginals,
        )

        plt.close(fig)

    if marginals_2D is not None:
        klds_ram = []

        for i, j in marginals_2D:
            counter = 0
            if flow_data is not None:
                fig, axs = plt.subplots(1, 2, figsize=(24, 10))

                plot_2D_free_energy(
                    flow_data[:, i],
                    flow_data[:, j],
                    weights=flow_data_weights,
                    ax=axs[0],
                    vmax=marginals_2D_vmax,
                    cmap=None,  # Use default
                    nbins=100,
                )
                axs[0].set_title("Flow")

                # axs[0].set_xlabel(f"dim {i}")
                # axs[0].set_ylabel(f"dim {j}")
                axs[0].set_xlabel(index_phi_psi_label_mapping[i] + f" (dim {i})")
                axs[0].set_ylabel(index_phi_psi_label_mapping[j] + f" (dim {j})")

                axs[0].set_xlim(data_range)
                axs[0].set_ylim(data_range)

                counter += 1

                if do_calculate_2D_marginal_kld:
                    kld_ram = calculate_2D_marginal_kld(
                        flow_data[:, [i, j]],
                        ground_truth_data[:, [i, j]],
                        (
                            flow_data_weights.view(-1)
                            if flow_data_weights is not None
                            else None
                        ),
                    )
                    klds_ram.append(kld_ram)

                    if report_wandb:
                        wandb.log(
                            {
                                f"KLD_{tag}_2Dmarginal_{i}_{j}": kld_ram,
                            },
                            step=current_i,
                        )
                    else:
                        print(f"KLD_{tag}_2Dmarginal_{i}_{j}: {kld_ram}")
            else:
                fig, axs = plt.subplots(1, 1, figsize=(12, 10))
                axs = [axs]

            plot_2D_free_energy(
                ground_truth_data[:, i],
                ground_truth_data[:, j],
                ax=axs[counter],
                vmax=marginals_2D_vmax,
                cmap=None,  # Use default
                nbins=100,
            )
            axs[counter].set_title("Ground truth")

            axs[counter].set_xlabel(index_phi_psi_label_mapping[i] + f" (dim {i})")
            axs[counter].set_ylabel(index_phi_psi_label_mapping[j] + f" (dim {j})")

            axs[counter].set_xlim(data_range)
            axs[counter].set_ylim(data_range)

            plt.tight_layout()

            log_figure(
                tag=f"{tag}_2Dmarginal_{i}_{j}",
                current_i=current_i,
                fig=plt.gcf(),
                write_pickle=True,
                report_wandb=report_wandb,
                output_dir=output_dir,
                dpi=dpi_2D_marginal,
            )

        if do_calculate_2D_marginal_kld and len(klds_ram) > 0 and flow_data is not None:
            mean_kld_ram = np.mean(klds_ram)

            if report_wandb:
                wandb.log(
                    {
                        f"KLD_{tag}_2Dmarginal_mean": mean_kld_ram,
                    },
                    step=current_i,
                )
            else:
                print(f"KLD_{tag}_2Dmarginal_mean: {mean_kld_ram}")
