import jax
from jax import numpy as jnp

@jax.jit
def cross_entropy(p, y):
    return -jnp.take(jnp.log(p.squeeze()), y.squeeze())