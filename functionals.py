import jax
from jax import numpy as jnp

from utils import flatten_dict

@jax.jit
def cross_entropy(params, p, y):
    return -jnp.take(jnp.log(p.squeeze()), y.squeeze())

@jax.jit
def weight_decay(params, p, y):
    return (norm(params, p=2.) ** 2.)

@jax.jit
def norm(params, p=2.):
    sqnorm = jnp.sum(flatten_dict(lambda x: (x**p).sum(), params))
    sqnorm = (sqnorm ** (1./p))
    return sqnorm