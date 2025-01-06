# Typed distributions for PyMC

This package adds explicit, typed arguments to pymc distributions.
It also removes some of the extra arguments that pymc distributions can take. I, at least, have never had to use `transforms`, `rng` `total_size` or `initval`. If you would like any of them added, just make an issue.

```python
import pymc as pm
import pymc_typed_distributions as ptd

# use ptd... to use a pymc distribution, with proper arguments
with pm.Model():
    mu = ptd.Normal(name="mu", mu=0, sigma=1)
    sigma = ptd.HalfNormal(name="sigma", sigma=1)
    likelihood = ptd.Normal(name="normal_test", mu=mu, sigma=sigma)
```

## Using pymc-typed-distributions
You can see that mu and sigma are arguments taken by `Normal()`. We also get proper types on the return object, not "partially unknown" ones.
![image](https://github.com/user-attachments/assets/4a7b0922-e394-45c6-9137-79d46ac92dcb)

## Using regular pymc
What are the arguments to `Normal`? Hard to tell!
![image](https://github.com/user-attachments/assets/fa53fef0-b19b-46e4-98c6-39a1d34df4c3)
