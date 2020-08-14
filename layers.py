import numpy

from jax import random as jrng
from jax import numpy as jnp
import jax

from jax import lax

from functools import partial

from utils import rand_string, split_and_sample

class Layer:
    def __init__(self, name=None):
        if name is None:
            self.name = F'Layer+{rand_string()}'
        else:
            self.name = name
            
        self.eval_ = False
    
    def __call__(self, p, x):
        return self.forward(p, x)
        
    def params(self):
        return None
    
    def init_params(self, rng):
        return rng, self.params()
    
    def forward(self, p, x):
        return x
    
    ''' not supported yet '''
    def train(self):
        self.eval_ = False
        
    ''' not supported yet '''
    def eval(self):
        self.eval_ = True
    
    
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
        self.weight = self.weight * (1./jnp.sqrt(self.weight.shape[0]))
        return rng, self.params()
    
    def forward(self, p, x):
        return jnp.dot(x, p['weight']) + p['bias']
    
    
class Conv2d(Layer):
    def __init__(self, k_height, k_width, d_in, d_out, name=None):
        super(Conv2d, self).__init__(name)
        
        if k_height is None:
            k_height = k_width
        
        self.weight = jnp.zeros((d_out, d_in, k_height, k_width))
        self.bias = jnp.zeros((d_out))
        
        if name is None:
            self.name = F'Conv2d+{rand_string()}'
        
    def params(self):
        return dict([('weight', self.weight), ('bias', self.bias)])
    
    def init_params(self, rng):
        rng, self.weight = split_and_sample(rng, self.weight.shape)
        self.weight = self.weight * (1./jnp.sqrt(self.weight.shape[1]*self.weight.shape[2]*self.weight.shape[3]))
        return rng, self.params()
    
    def forward(self, p, x):
        if len(x.shape) < len(p['weight'].shape):
            x = jnp.expand_dims(x, 0)
        return lax.conv(x, p['weight'], (1,1), 'SAME')# + p['bias'][None,:,None,None]
    
class SpatialPool2d(Layer):
    def __init__(self, name=None):
        super(SpatialPool2d, self).__init__(name)
        
        if name is None:
            self.name = F'SpatialPool2d+{rand_string()}'
        
    def forward(self, p, x):
        return x.max(-1).max(-1)
    
class Tanh(Layer):
    def __init__(self, name=None):
        super(Tanh, self).__init__(name)
        
        if name is None:
            self.name = F'Tanh+{rand_string()}'
            
    def forward(self, p, x):
        return jnp.tanh(x)
    
class ReLU(Layer):
    def __init__(self, name=None):
        super(ReLU, self).__init__(name)
        
        if name is None:
            self.name = F'ReLU+{rand_string()}'
            
    def forward(self, p, x):
        return jnp.maximum(0., x)
    
class Softmax(Layer):
    def __init__(self, name=None):
        super(Softmax, self).__init__(name)
        
        if name is None:
            self.name = F'Softmax+{rand_string()}'
            
    def forward(self, p, x):
        x_exp = jnp.exp(x)
        return x_exp / jnp.sum(x_exp)
    
