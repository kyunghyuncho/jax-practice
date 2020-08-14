from jax import random as jrng

import random
import string

def rand_string():
    return ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])

# @jax.jit
def split_and_sample(key, shape):
    key, subkey = jrng.split(key)
    val = jrng.normal(subkey, shape=shape)
    return key, val

