import pymc as pm
import pymc_typed_distributions as ptd


def test_binomial():
    with pm.Model():
        ptd.Binomial(name="binomial_test", n=10, p=0.7)


def test_beta_binomial():
    with pm.Model():
        ptd.BetaBinomial(name="beta_binom_test", alpha=2.0, beta=3.0, n=10)


def test_bernoulli():
    with pm.Model():
        ptd.Bernoulli(name="bernoulli_test", p=0.7)


def test_discrete_weibull():
    with pm.Model():
        ptd.DiscreteWeibull(name="discrete_weibull_test", q=0.6, beta=2.0)


def test_poisson():
    with pm.Model():
        ptd.Poisson(name="poisson_test", mu=3.0)


def test_negative_binomial():
    with pm.Model():
        ptd.NegativeBinomial(name="negative_binomial_test", mu=3.0, alpha=1.0)


def test_discrete_uniform():
    with pm.Model():
        ptd.DiscreteUniform(name="discrete_uniform_test", lower=0, upper=10)


def test_geometric():
    with pm.Model():
        ptd.Geometric(name="geometric_test", p=0.5)


def test_hypergeometric():
    with pm.Model():
        ptd.HyperGeometric(name="hypergeometric_test", N=100, k=30, n=10)


def test_categorical():
    with pm.Model():
        ptd.Categorical(name="categorical_test", p=[0.2, 0.5, 0.3])


def test_ordered_logistic():
    with pm.Model():
        ptd.OrderedLogistic(name="ordered_logistic_test", eta=0.5, cutpoints=[-1, 0, 1])


def test_ordered_probit():
    with pm.Model():
        ptd.OrderedProbit(name="ordered_probit_test", eta=0.5, cutpoints=[-1, 0, 1])
