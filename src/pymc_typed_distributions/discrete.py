import pymc as pm
from pymc.data import TensorVariable
from pymc.distributions.shape_utils import Dims
from numpy.typing import ArrayLike


def Binomial(
    name: str,
    n: ArrayLike | TensorVariable,
    p: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Binomial(name=name, n=n, p=p, dims=dims, observed=observed)


def BetaBinomial(
    name: str,
    alpha: ArrayLike | TensorVariable,
    beta: ArrayLike | TensorVariable,
    n: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.BetaBinomial(
        name=name, alpha=alpha, beta=beta, n=n, dims=dims, observed=observed
    )


def Bernoulli(
    name: str,
    p: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Bernoulli(name=name, p=p, dims=dims, observed=observed)


def DiscreteWeibull(
    name: str,
    q: ArrayLike | TensorVariable,
    beta: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.DiscreteWeibull(name=name, q=q, beta=beta, dims=dims, observed=observed)


def Poisson(
    name: str,
    mu: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Poisson(name=name, mu=mu, dims=dims, observed=observed)


def NegativeBinomial(
    name: str,
    mu: ArrayLike | TensorVariable,
    alpha: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.NegativeBinomial(
        name=name, mu=mu, alpha=alpha, dims=dims, observed=observed
    )


def DiscreteUniform(
    name: str,
    lower: ArrayLike | TensorVariable,
    upper: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.DiscreteUniform(
        name=name, lower=lower, upper=upper, dims=dims, observed=observed
    )


def Geometric(
    name: str,
    p: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Geometric(name=name, p=p, dims=dims, observed=observed)


def HyperGeometric(
    name: str,
    N: ArrayLike | TensorVariable,
    k: ArrayLike | TensorVariable,
    n: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.HyperGeometric(name=name, N=N, k=k, n=n, dims=dims, observed=observed)


def Categorical(
    name: str,
    p: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Categorical(name=name, p=p, dims=dims, observed=observed)


def OrderedLogistic(
    name: str,
    eta: ArrayLike | TensorVariable,
    cutpoints: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.OrderedLogistic(
        name=name, eta=eta, cutpoints=cutpoints, dims=dims, observed=observed
    )


def OrderedProbit(
    name: str,
    eta: ArrayLike | TensorVariable,
    cutpoints: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.OrderedProbit(
        name=name, eta=eta, cutpoints=cutpoints, dims=dims, observed=observed
    )
