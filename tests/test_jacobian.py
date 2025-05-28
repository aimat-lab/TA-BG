import bgflow as bg
import numpy as np
import torch
from bgflow import BoltzmannGenerator
from bgflow.nn.flow.sequential import SequentialFlow

from tests.utils import create_molecular_generator, create_toy_generator


def test_jacobian():
    torch.set_default_dtype(torch.float64)

    # Testing jacobian of toy generator with dims=4:
    _test_jacobian()

    # Testing jacobian of toy generator (flow only) with dims=10:
    generator, shape_info = create_toy_generator(dims=10)
    _test_jacobian(generator, shape_info, check_trafo=False)

    torch.set_default_dtype(torch.float32)

    # Testing jacobian of aldp generator (flow only):
    generator, shape_info, system = create_molecular_generator("aldp")
    _test_jacobian(generator, shape_info, system=system, check_trafo=False)

    # Testing jacobian of tetra generator (flow only):
    generator, shape_info, system = create_molecular_generator("tetra")
    _test_jacobian(generator, shape_info, system=system, check_trafo=False)

    # Testing jacobian of hexa generator (flow only):
    generator, shape_info, system = create_molecular_generator("hexa")
    _test_jacobian(generator, shape_info, system=system, check_trafo=False)


def _test_jacobian(
    generator: BoltzmannGenerator = None,
    shape_info: bg.ShapeDictionary = None,
    system=None,
    check_trafo=True,
):
    if generator is None:
        generator, shape_info = create_toy_generator()

    dims = [item[0] for item in shape_info.values()]

    ##### First test the jacobian of the coordinate trafo #####

    if check_trafo:
        trafo_dims = dims + [3, 3]  # Translation and rotation DOF

        coordinate_trafo = SequentialFlow(generator.flow._blocks[-1:])

        sample = generator.sample(1)
        IC_sample = coordinate_trafo.forward(sample, inverse=True)
        IC_sample = list(IC_sample)
        IC_sample[3] = IC_sample[3].view(1, 3)
        IC_sample = torch.cat(IC_sample[:-1], dim=1)
        IC_sample = IC_sample.view(-1)

        def trafo_func(x, return_jac=False):
            x = x.view(1, -1)
            x = torch.split(x, trafo_dims, dim=1)
            x = list(x)
            x[3] = x[3].view(1, 1, 3)

            output = coordinate_trafo.forward(*x, inverse=False)

            if not return_jac:
                return output[0].view(-1)
            else:
                return output[0].view(-1), output[1].view(-1).item()

        x = IC_sample.detach().requires_grad_(True)

        jacobian = torch.autograd.functional.jacobian(trafo_func, x)
        log_det_jacobian = torch.logdet(jacobian).item()

        bgflow_jac_det = trafo_func(x, return_jac=True)[1]

        assert np.isclose(log_det_jacobian, bgflow_jac_det, atol=1e-5)

    ##### Test the jacobian of the rest of the flow #####

    if system is not None:
        constraint_layers = 2 if system.system.getNumConstraints() > 0 else 0
    else:
        constraint_layers = 0

    IC_trafo = SequentialFlow(generator.flow._blocks[(-3 - constraint_layers) :])
    flow = SequentialFlow(generator.flow._blocks[: (-3 - constraint_layers)])

    sample = generator.sample(1)
    IC_sample = IC_trafo.forward(sample, inverse=True)
    IC_sample = list(IC_sample)[:-1]
    IC_sample = torch.cat(IC_sample, dim=1)
    IC_sample = IC_sample.view(-1)

    def flow_func(x, return_jac=False):
        x = x.view(1, -1)
        x = torch.split(x, dims, dim=1)

        output = flow.forward(*x, inverse=True)

        output = list(output)
        output_x = torch.cat(output[:-1], dim=1)

        if not return_jac:
            return output_x.view(-1)
        else:
            return output_x.view(-1), output[-1].view(-1).item()

    x = IC_sample.detach().requires_grad_(True)

    jacobian = torch.autograd.functional.jacobian(flow_func, x)
    log_det_jacobian = torch.logdet(jacobian).item()

    bgflow_jac_det = flow_func(x, return_jac=True)[1]

    assert np.isclose(log_det_jacobian, bgflow_jac_det, atol=1e-5)
