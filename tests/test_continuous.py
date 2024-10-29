import pymc as pm
import pymc_typed_distributions as ptd
import numpy as np


def test_uniform():
    with pm.Model():
        ptd.Uniform(name="uniform_test", lower=0.0, upper=1.0)


def test_halfflat():
    with pm.Model():
        ptd.HalfFlat(name="half_flat_test")


def test_truncated_normal():
    with pm.Model():
        ptd.TruncatedNormal(
            name="truncated_normal_test", mu=0.0, sigma=1.0, lower=0.0, upper=2.0
        )


def test_beta():
    with pm.Model():
        ptd.Beta(name="beta_test", alpha=2.0, beta=3.0)


def test_kumaraswamy():
    with pm.Model():
        ptd.Kumaraswamy(name="kumaraswamy_test", a=2.0, b=3.0)


def test_exponential():
    with pm.Model():
        ptd.Exponential(name="exponential_test", lam=1.0)


def test_laplace():
    with pm.Model():
        ptd.Laplace(name="laplace_test", mu=0.0, b=1.0)


def test_student_t():
    with pm.Model():
        ptd.StudentT(name="student_t_test", nu=5.0, mu=0.0, sigma=1.0)


def test_cauchy():
    with pm.Model():
        ptd.Cauchy(name="cauchy_test", alpha=0.0, beta=1.0)


def test_half_cauchy():
    with pm.Model():
        ptd.HalfCauchy(name="half_cauchy_test", beta=1.0)


def test_gamma():
    with pm.Model():
        ptd.Gamma(name="gamma_test", alpha=2.0, beta=1.0)


def test_weibull():
    with pm.Model():
        ptd.Weibull(name="weibull_test", alpha=2.0, beta=1.0)


def test_half_student_t():
    with pm.Model():
        ptd.HalfStudentT(name="half_student_t_test", nu=5.0, sigma=1.0)


def test_log_normal():
    with pm.Model():
        ptd.LogNormal(name="log_normal_test", mu=0.0, sigma=1.0)


def test_chi_squared():
    with pm.Model():
        ptd.ChiSquared(name="chi_squared_test", nu=2.0)


def test_half_normal():
    with pm.Model():
        ptd.HalfNormal(name="half_normal_test", sigma=1.0)


def test_wald():
    with pm.Model():
        ptd.Wald(name="wald_test", mu=1.0, lam=1.0, alpha=0.5)


def test_pareto():
    with pm.Model():
        ptd.Pareto(name="pareto_test", alpha=1.5, m=1.0)


def test_inverse_gamma():
    with pm.Model():
        ptd.InverseGamma(name="inverse_gamma_test", alpha=2.0, beta=1.0)


def test_ex_gaussian():
    with pm.Model():
        ptd.ExGaussian(name="ex_gaussian_test", mu=0.0, sigma=1.0, nu=2.0)


def test_von_mises():
    with pm.Model():
        ptd.VonMises(name="von_mises_test", mu=0.0, kappa=1.0)


def test_skew_normal():
    with pm.Model():
        ptd.SkewNormal(name="skew_normal_test", mu=0.0, sigma=1.0, alpha=2.0)


def test_triangular():
    with pm.Model():
        ptd.Triangular(name="triangular_test", lower=0.0, upper=1.0, c=0.5)


def test_gumbel():
    with pm.Model():
        ptd.Gumbel(name="gumbel_test", mu=0.0, beta=1.0)


def test_logistic():
    with pm.Model():
        ptd.Logistic(name="logistic_test", mu=0.0, s=1.0)


def test_logit_normal():
    with pm.Model():
        ptd.LogitNormal(name="logit_normal_test", mu=0.0, sigma=1.0)


def test_interpolated():
    with pm.Model():
        ptd.Interpolated(
            name="interpolated_test",
            x_points=np.array([0, 1, 2]),
            pdf_points=np.array([0.1, 0.4, 0.5]),
        )


def test_rice():
    with pm.Model():
        ptd.Rice(name="rice_test", b=0.5, sigma=1.0)


def test_moyal():
    with pm.Model():
        ptd.Moyal(name="moyal_test", mu=0.0, sigma=1.0)


def test_asymmetric_laplace():
    with pm.Model():
        ptd.AsymmetricLaplace(name="asymmetric_laplace_test", b=1.0, kappa=2.0, mu=0.0)


def test_polya_gamma():
    with pm.Model():
        ptd.PolyaGamma(name="polya_gamma_test", h=1.0, z=2.0)


def test_skew_student_t():
    with pm.Model():
        ptd.SkewStudentT(
            name="skew_student_t_test",
            a=5.0,
            b=0.0,
            mu=0.5,
        )
