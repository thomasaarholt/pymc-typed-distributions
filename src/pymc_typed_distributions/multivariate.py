import pymc as pm
from pymc.distributions.shape_utils import Dims
from numpy.typing import ArrayLike

from .types import TensorVariable


def MvNormal(
    name: str,
    mu: ArrayLike | TensorVariable,
    cov: ArrayLike | TensorVariable | None = None,
    tau: ArrayLike | TensorVariable | None = None,
    chol: ArrayLike | TensorVariable | None = None,
    lower: bool = True,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.MvNormal(
        name=name,
        mu=mu,
        cov=cov,
        tau=tau,
        chol=chol,
        lower=lower,
        dims=dims,
        observed=observed,
    )


def MvStudentT(
    name: str,
    nu: ArrayLike | TensorVariable,
    mu: ArrayLike | TensorVariable,
    scale: ArrayLike | TensorVariable,
    tau: ArrayLike | TensorVariable | None = None,
    chol: ArrayLike | TensorVariable | None = None,
    lower: bool = True,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.MvStudentT(
        name=name,
        nu=nu,
        mu=mu,
        scale=scale,
        tau=tau,
        chol=chol,
        dims=dims,
        observed=observed,
    )


def Dirichlet(
    name: str,
    a: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Dirichlet(name=name, a=a, dims=dims, observed=observed)


def Multinomial(
    name: str,
    n: ArrayLike | TensorVariable,
    p: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Multinomial(name=name, n=n, p=p, dims=dims, observed=observed)


def DirichletMultinomial(
    name: str,
    n: ArrayLike | TensorVariable,
    p: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.DirichletMultinomial(name=name, n=n, p=p, dims=dims, observed=observed)
