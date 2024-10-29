from collections.abc import Sequence
import pymc as pm
from pymc.data import TensorVariable
from pymc.distributions.shape_utils import Dims
from numpy.typing import ArrayLike


def HurdlePoisson(
    name: str,
    psi: float | TensorVariable,
    mu: float | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.HurdlePoisson(name=name, psi=psi, mu=mu, dims=dims, observed=observed)


def HurdleNegativeBinomial(
    name: str,
    psi: float | TensorVariable,
    mu: float | TensorVariable,
    alpha: float | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.HurdleNegativeBinomial(
        name=name, psi=psi, mu=mu, alpha=alpha, dims=dims, observed=observed
    )


def HurdleGamma(
    name: str,
    psi: float | TensorVariable,
    alpha: float | TensorVariable,
    beta: float | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.HurdleGamma(
        name=name, psi=psi, alpha=alpha, beta=beta, dims=dims, observed=observed
    )


def HurdleLogNormal(
    name: str,
    psi: float | TensorVariable,
    mu: float | TensorVariable,
    sigma: float | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.HurdleLogNormal(
        name=name,
        psi=psi,
        mu=mu,  # type:ignore # should be float
        sigma=sigma,
        dims=dims,
        observed=observed,
    )  # type: ignore


def Mixture(
    name: str,
    w: Sequence[float] | TensorVariable,
    comp_dists: Sequence[TensorVariable],
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Mixture(
        name=name, w=w, comp_dists=comp_dists, dims=dims, observed=observed
    )


def NormalMixture(
    name: str,
    w: Sequence[float] | TensorVariable,
    mu: Sequence[float] | TensorVariable,
    sigma: Sequence[float] | TensorVariable | None = None,
    tau: Sequence[float] | TensorVariable | None = None,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.NormalMixture(
        name=name, w=w, mu=mu, sigma=sigma, tau=tau, dims=dims, observed=observed
    )


def ZeroInflatedPoisson(
    name: str,
    psi: float | TensorVariable,
    mu: float | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.ZeroInflatedPoisson(
        name=name, psi=psi, mu=mu, dims=dims, observed=observed
    )


def ZeroInflatedBinomial(
    name: str,
    psi: float | TensorVariable,
    n: int | TensorVariable,
    p: float | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.ZeroInflatedBinomial(
        name=name, psi=psi, n=n, p=p, dims=dims, observed=observed
    )


def ZeroInflatedNegativeBinomial(
    name: str,
    psi: float | TensorVariable,
    mu: float | TensorVariable,
    alpha: float | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.ZeroInflatedNegativeBinomial(
        name=name, psi=psi, mu=mu, alpha=alpha, dims=dims, observed=observed
    )
