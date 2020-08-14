from jax import random as jrng
import jax

import random
import string

def rand_string():
    return ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])

# @jax.jit
def split_and_sample(key, shape):
    key, subkey = jrng.split(key)
    val = jrng.normal(subkey, shape=shape)
    return key, val

def apply_dict(func, from_, to_):
    for kk, vv in to_.items():
        if type(vv) == dict:
            apply_dict(func, from_[kk], vv)
        else:
            to_[kk] = func(from_[kk], to_[kk])

def flatten_dict(item_func, dict_):
    items = []
    for kk, vv in dict_.items():
        if type(vv) == dict:
            items = items + flatten_dict(item_func, vv)
        else:
            items.append(item_func(vv))
    return items

