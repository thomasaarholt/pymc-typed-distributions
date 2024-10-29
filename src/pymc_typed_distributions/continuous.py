import pymc as pm
from pymc.data import TensorVariable
from pymc.distributions.shape_utils import Dims
from numpy.typing import ArrayLike

pm.Mixture


def Uniform(
    name: str,
    lower: float,
    upper: float,
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
    mu: float,
    sigma: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Normal(name=name, mu=mu, sigma=sigma, dims=dims, observed=observed)


def TruncatedNormal(
    name: str,
    mu: float,
    sigma: float,
    lower: float,
    upper: float,
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
    alpha: float,
    beta: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Beta(name=name, alpha=alpha, beta=beta, dims=dims, observed=observed)


def Kumaraswamy(
    name: str,
    a: float,
    b: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Kumaraswamy(name=name, a=a, b=b, dims=dims, observed=observed)


def Exponential(
    name: str, lam: float, dims: Dims | None = None, observed: ArrayLike | None = None
) -> TensorVariable:
    return pm.Exponential(name=name, lam=lam, dims=dims, observed=observed)


def Laplace(
    name: str,
    mu: float,
    b: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Laplace(name=name, mu=mu, b=b, dims=dims, observed=observed)


def StudentT(
    name: str,
    nu: float,
    mu: float,
    sigma: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.StudentT(
        name=name, nu=nu, mu=mu, sigma=sigma, dims=dims, observed=observed
    )


def Cauchy(
    name: str,
    alpha: float,
    beta: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Cauchy(name=name, alpha=alpha, beta=beta, dims=dims, observed=observed)


def HalfCauchy(
    name: str, beta: float, dims: Dims | None = None, observed: ArrayLike | None = None
) -> TensorVariable:
    return pm.HalfCauchy(name=name, beta=beta, dims=dims, observed=observed)


def Gamma(
    name: str,
    alpha: float,
    beta: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Gamma(name=name, alpha=alpha, beta=beta, dims=dims, observed=observed)


def Weibull(
    name: str,
    alpha: float,
    beta: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Weibull(name=name, alpha=alpha, beta=beta, dims=dims, observed=observed)


def HalfStudentT(
    name: str,
    nu: float,
    sigma: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.HalfStudentT(name=name, nu=nu, sigma=sigma, dims=dims, observed=observed)


def LogNormal(
    name: str,
    mu: float,
    sigma: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.LogNormal(name=name, mu=mu, sigma=sigma, dims=dims, observed=observed)


def ChiSquared(
    name: str, nu: float, dims: Dims | None = None, observed: ArrayLike | None = None
) -> TensorVariable:
    return pm.ChiSquared(name=name, nu=nu, dims=dims, observed=observed)


def HalfNormal(
    name: str, sigma: float, dims: Dims | None = None, observed: ArrayLike | None = None
) -> TensorVariable:
    return pm.HalfNormal(name=name, sigma=sigma, dims=dims, observed=observed)


def Wald(
    name: str,
    mu: float,
    lam: float,
    alpha: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Wald(name=name, mu=mu, lam=lam, alpha=alpha, dims=dims, observed=observed)


def Pareto(
    name: str,
    alpha: float,
    m: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Pareto(name=name, alpha=alpha, m=m, dims=dims, observed=observed)


def InverseGamma(
    name: str,
    alpha: float,
    beta: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.InverseGamma(
        name=name, alpha=alpha, beta=beta, dims=dims, observed=observed
    )


def ExGaussian(
    name: str,
    mu: float,
    sigma: float,
    nu: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.ExGaussian(
        name=name, mu=mu, sigma=sigma, nu=nu, dims=dims, observed=observed
    )


def VonMises(
    name: str,
    mu: float,
    kappa: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.VonMises(name=name, mu=mu, kappa=kappa, dims=dims, observed=observed)


def SkewNormal(
    name: str,
    mu: float,
    sigma: float,
    alpha: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.SkewNormal(
        name=name, mu=mu, sigma=sigma, alpha=alpha, dims=dims, observed=observed
    )


def Triangular(
    name: str,
    lower: float,
    upper: float,
    c: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Triangular(
        name=name, lower=lower, upper=upper, c=c, dims=dims, observed=observed
    )


def Gumbel(
    name: str,
    mu: float,
    beta: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Gumbel(name=name, mu=mu, beta=beta, dims=dims, observed=observed)


def Logistic(
    name: str,
    mu: float,
    s: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Logistic(name=name, mu=mu, s=s, dims=dims, observed=observed)


def LogitNormal(
    name: str,
    mu: float,
    sigma: float,
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
    b: float,
    sigma: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Rice(name=name, b=b, sigma=sigma, dims=dims, observed=observed)


def Moyal(
    name: str,
    mu: float,
    sigma: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.Moyal(name=name, mu=mu, sigma=sigma, dims=dims, observed=observed)


def AsymmetricLaplace(
    name: str,
    b: float,
    kappa: float,
    mu: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.AsymmetricLaplace(
        name=name, b=b, kappa=kappa, mu=mu, dims=dims, observed=observed
    )


def PolyaGamma(
    name: str,
    h: float,
    z: float,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.PolyaGamma(name=name, h=h, z=z, dims=dims, observed=observed)


def SkewStudentT(
    name: str,
    a: float | TensorVariable,
    b: float | TensorVariable,
    mu: float | TensorVariable = 0,
    sigma: float | TensorVariable | None = None,
    lam: float | TensorVariable | None = None,
    dims: Dims | None = None,
    observed: ArrayLike | None = None,
) -> TensorVariable:
    return pm.SkewStudentT(
        name=name, a=a, b=b, mu=mu, sigma=sigma, lam=lam, dims=dims, observed=observed
    )
