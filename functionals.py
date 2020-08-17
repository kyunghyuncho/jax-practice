import jax
from jax import numpy as jnp

from utils import flatten_dict, map_dict

@jax.jit
def cross_entropy(params, buffers, p, y):
    return -jnp.take(jnp.log(p.squeeze()), y.squeeze())

@jax.jit
def weight_decay(params, buffers, p, y):
    return (norm(params, p=2.) ** 2.)

@jax.jit
def norm(params, p=2.):
    sqnorm = jnp.sum(flatten_dict(lambda x: (x**p).sum(), params))
    sqnorm = (sqnorm ** (1./p))
    return sqnorm

def clip_norm(params, thr=1., p=2.):
    pnorm = norm(params, p=p)
    if pnorm <= thr:
        return params
    return map_dict(lambda x: x / pnorm * thr, params)

@jax.jit
def relu(x):
    return jnp.maximum(0., x)

@jax.jit
def leaky_relu(x, alpha=0.001):
    return jnp.maximum(0., x) - alpha * jnp.maximum(0., -x)