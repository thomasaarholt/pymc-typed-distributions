from collections.abc import Sequence
import pymc as pm
from pymc.data import TensorVariable
from pymc.distributions.shape_utils import Dims
from numpy.typing import ArrayLike


def Binomial(
    name: str,
    n: int,
    p: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Binomial(name=name, n=n, p=p, dims=dims, observed=observed)


def BetaBinomial(
    name: str,
    alpha: float,
    beta: float,
    n: int,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.BetaBinomial(
        name=name, alpha=alpha, beta=beta, n=n, dims=dims, observed=observed
    )


def Bernoulli(
    name: str, p: float, dims: Dims | None = None, observed: ArrayLike | None = None
) -> TensorVariable:
    return pm.Bernoulli(name=name, p=p, dims=dims, observed=observed)


def DiscreteWeibull(
    name: str,
    q: float,
    beta: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.DiscreteWeibull(name=name, q=q, beta=beta, dims=dims, observed=observed)


def Poisson(
    name: str, mu: float, dims: Dims | None = None, observed: ArrayLike | None = None
) -> TensorVariable:
    return pm.Poisson(name=name, mu=mu, dims=dims, observed=observed)


def NegativeBinomial(
    name: str,
    mu: float,
    alpha: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.NegativeBinomial(
        name=name, mu=mu, alpha=alpha, dims=dims, observed=observed
    )


def DiscreteUniform(
    name: str,
    lower: int,
    upper: int,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.DiscreteUniform(
        name=name, lower=lower, upper=upper, dims=dims, observed=observed
    )


def Geometric(
    name: str, p: float, dims: Dims | None = None, observed: ArrayLike | None = None
) -> TensorVariable:
    return pm.Geometric(name=name, p=p, dims=dims, observed=observed)


def HyperGeometric(
    name: str,
    N: int,
    k: int,
    n: int,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.HyperGeometric(name=name, N=N, k=k, n=n, dims=dims, observed=observed)


def Categorical(
    name: str,
    p: Sequence[float],
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Categorical(name=name, p=p, dims=dims, observed=observed)


def OrderedLogistic(
    name: str,
    eta: float,
    cutpoints: list[float] | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.OrderedLogistic(
        name=name, eta=eta, cutpoints=cutpoints, dims=dims, observed=observed
    )


def OrderedProbit(
    name: str,
    eta: float,
    cutpoints: list[float] | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.OrderedProbit(
        name=name, eta=eta, cutpoints=cutpoints, dims=dims, observed=observed
    )
