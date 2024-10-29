# Typed distributions for PyMC

This package simply provides typed arguments, with all commonly used arguments, to pymc.

```python

def normal():
    with pm.Model():
        mu = ptd.normal(name="mu", mu=0, sigma=1)
        sigma = ptd.HalfNormal(name="sigma", sigma=1)
        likelihood = ptd.Normal(name="normal_test", mu=mu, sigma=sigma)

```

