import pymc as pm
from pymc.data import TensorVariable
from pymc.distributions.shape_utils import Dims
from numpy.typing import ArrayLike


def Uniform(
    name: str,
    lower: ArrayLike | TensorVariable,
    upper: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Uniform(name=name, lower=lower, upper=upper, dims=dims, observed=observed)


def Flat(
    name: str, dims: Dims | None = None, observed: ArrayLike | None = None
) -> TensorVariable:
    return pm.Flat(name=name, dims=dims, observed=observed)


def HalfFlat(
    name: str, dims: Dims | None = None, observed: ArrayLike | None = None
) -> TensorVariable:
    return pm.HalfFlat(name=name, dims=dims, observed=observed)


def Normal(
    name: str,
    mu: ArrayLike | TensorVariable,
    sigma: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Normal(name=name, mu=mu, sigma=sigma, dims=dims, observed=observed)


def TruncatedNormal(
    name: str,
    mu: ArrayLike | TensorVariable,
    sigma: ArrayLike | TensorVariable,
    lower: ArrayLike | TensorVariable,
    upper: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.TruncatedNormal(
        name=name,
        mu=mu,
        sigma=sigma,
        lower=lower,
        upper=upper,
        dims=dims,
        observed=observed,
    )


def Beta(
    name: str,
    alpha: ArrayLike | TensorVariable,
    beta: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Beta(name=name, alpha=alpha, beta=beta, dims=dims, observed=observed)


def Kumaraswamy(
    name: str,
    a: ArrayLike | TensorVariable,
    b: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Kumaraswamy(name=name, a=a, b=b, dims=dims, observed=observed)


def Exponential(
    name: str,
    lam: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Exponential(name=name, lam=lam, dims=dims, observed=observed)


def Laplace(
    name: str,
    mu: ArrayLike | TensorVariable,
    b: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Laplace(name=name, mu=mu, b=b, dims=dims, observed=observed)


def StudentT(
    name: str,
    nu: ArrayLike | TensorVariable,
    mu: ArrayLike | TensorVariable,
    sigma: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.StudentT(
        name=name, nu=nu, mu=mu, sigma=sigma, dims=dims, observed=observed
    )


def Cauchy(
    name: str,
    alpha: ArrayLike | TensorVariable,
    beta: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Cauchy(name=name, alpha=alpha, beta=beta, dims=dims, observed=observed)


def HalfCauchy(
    name: str,
    beta: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.HalfCauchy(name=name, beta=beta, dims=dims, observed=observed)


def Gamma(
    name: str,
    alpha: ArrayLike | TensorVariable,
    beta: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Gamma(name=name, alpha=alpha, beta=beta, dims=dims, observed=observed)


def Weibull(
    name: str,
    alpha: ArrayLike | TensorVariable,
    beta: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Weibull(name=name, alpha=alpha, beta=beta, dims=dims, observed=observed)


def HalfStudentT(
    name: str,
    nu: ArrayLike | TensorVariable,
    sigma: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.HalfStudentT(name=name, nu=nu, sigma=sigma, dims=dims, observed=observed)


def LogNormal(
    name: str,
    mu: ArrayLike | TensorVariable,
    sigma: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.LogNormal(name=name, mu=mu, sigma=sigma, dims=dims, observed=observed)


def ChiSquared(
    name: str,
    nu: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.ChiSquared(name=name, nu=nu, dims=dims, observed=observed)


def HalfNormal(
    name: str,
    sigma: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.HalfNormal(name=name, sigma=sigma, dims=dims, observed=observed)


def Wald(
    name: str,
    mu: ArrayLike | TensorVariable,
    lam: ArrayLike | TensorVariable,
    alpha: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Wald(name=name, mu=mu, lam=lam, alpha=alpha, dims=dims, observed=observed)


def Pareto(
    name: str,
    alpha: ArrayLike | TensorVariable,
    m: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Pareto(name=name, alpha=alpha, m=m, dims=dims, observed=observed)


def InverseGamma(
    name: str,
    alpha: ArrayLike | TensorVariable,
    beta: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.InverseGamma(
        name=name, alpha=alpha, beta=beta, dims=dims, observed=observed
    )


def ExGaussian(
    name: str,
    mu: ArrayLike | TensorVariable,
    sigma: ArrayLike | TensorVariable,
    nu: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.ExGaussian(
        name=name, mu=mu, sigma=sigma, nu=nu, dims=dims, observed=observed
    )


def VonMises(
    name: str,
    mu: ArrayLike | TensorVariable,
    kappa: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.VonMises(name=name, mu=mu, kappa=kappa, dims=dims, observed=observed)


def SkewNormal(
    name: str,
    mu: ArrayLike | TensorVariable,
    sigma: ArrayLike | TensorVariable,
    alpha: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.SkewNormal(
        name=name, mu=mu, sigma=sigma, alpha=alpha, dims=dims, observed=observed
    )


def Triangular(
    name: str,
    lower: ArrayLike | TensorVariable,
    upper: ArrayLike | TensorVariable,
    c: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Triangular(
        name=name, lower=lower, upper=upper, c=c, dims=dims, observed=observed
    )


def Gumbel(
    name: str,
    mu: ArrayLike | TensorVariable,
    beta: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Gumbel(name=name, mu=mu, beta=beta, dims=dims, observed=observed)


def Logistic(
    name: str,
    mu: ArrayLike | TensorVariable,
    s: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Logistic(name=name, mu=mu, s=s, dims=dims, observed=observed)


def LogitNormal(
    name: str,
    mu: ArrayLike | TensorVariable,
    sigma: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.LogitNormal(name=name, mu=mu, sigma=sigma, dims=dims, observed=observed)


def Interpolated(
    name: str,
    x_points: ArrayLike,
    pdf_points: ArrayLike,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Interpolated(
        name=name,
        x_points=x_points,
        pdf_points=pdf_points,
        dims=dims,
        observed=observed,
    )


def Rice(
    name: str,
    b: ArrayLike | TensorVariable,
    sigma: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Rice(name=name, b=b, sigma=sigma, dims=dims, observed=observed)


def Moyal(
    name: str,
    mu: ArrayLike | TensorVariable,
    sigma: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Moyal(name=name, mu=mu, sigma=sigma, dims=dims, observed=observed)


def AsymmetricLaplace(
    name: str,
    b: ArrayLike | TensorVariable,
    kappa: ArrayLike | TensorVariable,
    mu: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.AsymmetricLaplace(
        name=name, b=b, kappa=kappa, mu=mu, dims=dims, observed=observed
    )


def PolyaGamma(
    name: str,
    h: ArrayLike | TensorVariable,
    z: ArrayLike | TensorVariable,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.PolyaGamma(name=name, h=h, z=z, dims=dims, observed=observed)


def SkewStudentT(
    name: str,
    a: ArrayLike | TensorVariable,
    b: ArrayLike | TensorVariable,
    mu: ArrayLike | TensorVariable = 0,
    sigma: ArrayLike | TensorVariable | None = None,
    lam: ArrayLike | TensorVariable | None = None,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.SkewStudentT(
        name=name, a=a, b=b, mu=mu, sigma=sigma, lam=lam, dims=dims, observed=observed
    )
