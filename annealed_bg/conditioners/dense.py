from typing import Sequence

import numpy as np
import torch


class DenseNet(torch.nn.Module):
    def __init__(
        self,
        n_units: list,
        activation=torch.nn.ReLU(),
        weight_scale: float = 1.0,
        bias_scale: float = 0.0,
        add_skip_connection: bool = False,
        concatenate_context: bool = False,
    ):
        """
        Simple multi-layer perceptron with optional skip connection.

        Args:
            n_units (List / Tuple of integers): Number of units in each layer.
            activation: Non-linearity
            weight_scale (float): Scaling factor for the weights.
            bias_scale (float): Scaling factor for the biases.
            add_skip_connection (bool): If True, adds a skip connection from first hidden layer to last hidden layer.
            concatenate_context (bool): If True, concatenate context to input before passing through network.
        """

        super().__init__()

        self.add_skip_connection = add_skip_connection
        self.concatenate_context = concatenate_context
        self.activation = activation

        dims_in = n_units[:-1]
        dims_out = n_units[1:]

        self.first_layer = torch.nn.Linear(dims_in[0], dims_out[0])
        self.first_layer.weight.data *= weight_scale
        if bias_scale > 0.0:
            self.first_layer.bias.data = (
                torch.Tensor(self.first_layer.bias.data).uniform_() * bias_scale
            )

        layers = []
        for i, (dim_in, dim_out) in enumerate(zip(dims_in[1:], dims_out[1:]), start=1):
            layers.append(torch.nn.Linear(dim_in, dim_out))
            layers[-1].weight.data *= weight_scale
            if bias_scale > 0.0:
                layers[-1].bias.data = (
                    torch.Tensor(layers[-1].bias.data).uniform_() * bias_scale
                )
            if i < len(n_units) - 2:
                if activation is not None:
                    layers.append(activation)

        self.hidden_layers = torch.nn.Sequential(*layers[:-1])
        self.final_layer = layers[-1]

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None):
        if context is not None and self.concatenate_context:
            x = torch.cat([x, context], dim=-1)

        first_out = self.first_layer(x)
        hidden_output = self.hidden_layers(first_out)

        if self.add_skip_connection:
            if first_out.shape[-1] == hidden_output.shape[-1]:
                hidden_output = hidden_output + self.activation(first_out)
            else:
                raise ValueError("Skip connection dimensions do not match.")

        output = self.final_layer(hidden_output)
        return output


def _make_dense_conditioner(
    dim_in: int,
    dim_out: int,
    context_dims: int = 0,
    init_spline_identity: bool = True,
    init_zeros: bool = False,
    hidden: Sequence = (128, 128),
    activation=torch.nn.ReLU(),
    add_skip_connection: bool = False,
    **kwargs,
):
    assert not (
        init_spline_identity and init_zeros
    ), "Cannot initialize both to zeros and spline identity."

    net = DenseNet(
        [dim_in + context_dims, *hidden, dim_out],
        activation=activation,
        add_skip_connection=add_skip_connection,
        concatenate_context=(context_dims > 0) if context_dims is not None else False,
    )

    if init_spline_identity:
        DEFAULT_MIN_DERIVATIVE = 1e-3
        torch.nn.init.constant_(net.final_layer.weight, 0.0)
        torch.nn.init.constant_(
            net.final_layer.bias, np.log(np.exp(1 - DEFAULT_MIN_DERIVATIVE) - 1)
        )
    elif init_zeros:
        torch.nn.init.zeros_(net.final_layer.weight)
        torch.nn.init.zeros_(net.final_layer.bias)

    return net
