import numpy

from jax import random as jrng
from jax import numpy as jnp
import jax

from functools import partial

from utils import rand_string, split_and_sample

class Layer:
    def __init__(self, name=None):
        if name is None:
            self.name = F'Layer+{rand_string()}'
        else:
            self.name = name
    
    def __call__(self, p, x):
        return self.forward(p, x)
        
    def params(self):
        return None
    
    def init_params(self, rng):
        return rng, self.params()
    
    def forward(self, p, x):
        return x
    
    
class Linear(Layer):
    def __init__(self, d_in, d_out, name=None):
        super(Linear, self).__init__(name)
        
        self.weight = jnp.zeros((d_in, d_out))
        self.bias = jnp.zeros((d_out))
        
        if name is None:
            self.name = F'Linear+{rand_string()}'
        
    def params(self):
        return dict([('weight', self.weight), ('bias', self.bias)])
    
    def init_params(self, rng):
        rng, self.weight = split_and_sample(rng, self.weight.shape)
        return rng, self.params()
    
    def forward(self, p, x):
        return jnp.dot(x, p['weight']) + p['bias']
    
class Tanh(Layer):
    def __init__(self, name=None):
        super(Tanh, self).__init__(name)
        
        if name is None:
            self.name = F'Tanh+{rand_string()}'
            
    def forward(self, p, x):
        return jnp.tanh(x)
    
class Softmax(Layer):
    def __init__(self, name=None):
        super(Softmax, self).__init__(name)
        
        if name is None:
            self.name = F'Softmax+{rand_string()}'
            
    def forward(self, p, x):
        x_exp = jnp.exp(x)
        return x_exp / jnp.sum(x_exp)
    
