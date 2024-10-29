import pymc as pm
import pymc_typed_distributions as ptd


def test_mixture():
    with pm.Model():
        w = [0.5, 0.5]
        comp_dists = [pm.Normal.dist(mu=0, sigma=1), pm.Normal.dist(mu=2, sigma=3)]
        ptd.Mixture(name="mixture_test", w=w, comp_dists=comp_dists)


def test_normal_mixture():
    with pm.Model():
        w = [0.3, 0.7]
        ptd.NormalMixture(name="normal_mixture_test", w=w, mu=[0, 5], sigma=[1, 2])


def test_zero_inflated_poisson():
    with pm.Model():
        ptd.ZeroInflatedPoisson(name="zip_test", psi=0.5, mu=3)


def test_zero_inflated_binomial():
    with pm.Model():
        ptd.ZeroInflatedBinomial(name="zib_test", psi=0.4, n=10, p=0.7)


def test_zero_inflated_negative_binomial():
    with pm.Model():
        ptd.ZeroInflatedNegativeBinomial(name="zinb_test", psi=0.4, mu=3, alpha=2)


def test_hurdle_poisson():
    with pm.Model():
        ptd.HurdlePoisson(name="hurdle_poisson_test", psi=0.4, mu=3)


def test_hurdle_negative_binomial():
    with pm.Model():
        ptd.HurdleNegativeBinomial(name="hurdle_neg_binom_test", psi=0.5, mu=4, alpha=1)


def test_hurdle_gamma():
    with pm.Model():
        ptd.HurdleGamma(name="hurdle_gamma_test", psi=0.5, alpha=2, beta=1)


def test_hurdle_lognormal():
    with pm.Model():
        ptd.HurdleLogNormal(name="hurdle_lognormal_test", psi=0.6, mu=1, sigma=2)
