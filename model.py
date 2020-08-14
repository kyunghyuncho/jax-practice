import numpy

from jax import random as jrng
from jax import numpy as jnp
import jax

from functools import partial

from utils import rand_string, split_and_sample

class Model:
    def __init__(self, rng, layers, loss=None, name=None):        
        if name is None:
            name = F'Model+{rand_string()}'
            
        self.layers = layers
        
        if type(loss) is not list:
            self.loss = [(loss, 1.)]
        else:
            self.loss = loss
            
        self.params = dict()
        for ll in self.layers:
            rng, pp = ll.init_params(rng)
            if pp is not None:
                self.params[ll.name] = pp
        self.params_values, self.params_tree = jax.tree_flatten(self.params)

    @partial(jax.jit, static_argnums=(0,))
    def forward_(self, p, x):
        h = x
        for ll in self.layers:
            h = ll(None if ll.name not in p else p[ll.name], h)
        return h    
    
    @partial(jax.jit, static_argnums=(0,))
    def loss_(self, p, x, y):
        def dummy(mymodel, params, x, y):
            total_l = 0.
            for ll in mymodel.loss:
                total_l = total_l + ll[1] * ll[0](x, y)
            return total_l
        return jax.vmap(dummy, in_axes=(None,None,0,0))(self, self.params, self.forward_(p, x), y).mean()
    
    def forward(self, x, single=False):
        if single:
            return self.forward_(self.params, x)
        
        def dummy(mymodel, params, x):
            return mymodel.forward_(params, x)
        return jax.vmap(dummy, in_axes=(None, None, 0))(self, self.params, x)
    
    def loss_grad(self, x, y):
        return self.loss_(self.params, x, y), jax.grad(self.loss_)(self.params, x, y)