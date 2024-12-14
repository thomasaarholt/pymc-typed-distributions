import pymc as pm
from numpy.typing import ArrayLike
from pymc.distributions.shape_utils import Dims

from .types import TensorVariable


def Deterministic(
    name: str,
    var: ArrayLike | TensorVariable,
    model: pm.Model | None = None,
    dims: Dims | None = None,
) -> TensorVariable:
    return pm.Deterministic(name=name, var=var, model=model, dims=dims)
