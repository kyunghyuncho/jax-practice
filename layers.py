import numpy

from jax import random as jrng
from jax import numpy as jnp
import jax.experimental.stax as jexp
import jax

from jax import lax

from functools import partial

from utils import rand_string, split_and_sample
import functionals as F

class Layer:
    def __init__(self, name=None):
        if name is None:
            self.name = F'Layer+{rand_string()}'
        else:
            self.name = name
            
        self.eval_ = False
    
    def __call__(self, p, b, x):
        return self.forward(p, b, x)
        
    def params(self):
        return None
    
    def init_params(self, rng):
        return rng, self.params()
    
    def buffers(self):
        return None
    
    # p: params, b: buffers
    def forward(self, p, b, x):
        return x
    
    # p: params, b: buffers
    def forward_eval(self, p, b, x):
        return self.forward(p, b, x)    
    
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
    
    @partial(jax.jit, static_argnums=(0,))
    def forward(self, p, b, x):
        return jnp.dot(x, p['weight']) + p['bias']
    
class Conv2d(Layer):
    def __init__(self, k_height, k_width, d_in, d_out, name=None, mode="SAME"):
        super(Conv2d, self).__init__(name)
        
        if k_height is None:
            k_height = k_width
        
        self.weight = jnp.zeros((d_out, d_in, k_height, k_width))
        self.bias = jnp.zeros((d_out))
        
        self.mode = mode
        
        if name is None:
            self.name = F'Conv2d+{rand_string()}'
        
    def params(self):
        return dict([('weight', self.weight), ('bias', self.bias)])
    
    def init_params(self, rng):
        rng, self.weight = split_and_sample(rng, self.weight.shape)
        self.weight = self.weight * (1./jnp.sqrt(self.weight.shape[1]*self.weight.shape[2]*self.weight.shape[3]))
        return rng, self.params()
    
    def forward(self, p, b, x):
        if len(x.shape) < len(p['weight'].shape):
            x = jnp.expand_dims(x, 0)
        return lax.conv(x, p['weight'], (1,1), self.mode) + p['bias'][None,:,None,None]

''' this is a fake but simpler residual layer '''
class FakeResConv2d(Layer):
    def __init__(self, k_height, k_width, d_in, d_out, name=None, mode="SAME"):
        super(FakeResConv2d, self).__init__(name)
        
        assert d_in == d_out
        assert mode == "SAME"
        
        if k_height is None:
            k_height = k_width
        
        self.weight1 = jnp.zeros((d_out, d_in, 1, 1))
        self.bias1 = jnp.zeros(d_out)
        self.weight2 = jnp.zeros((d_out, d_in, k_height, k_width))
        self.bias2 = jnp.zeros((d_out))
        self.weight3 = jnp.zeros((d_out, d_in, 1, 1))
        self.bias3 = jnp.zeros(d_out)
        
        self.mode = mode
        
        if name is None:
            self.name = F'Conv2d+{rand_string()}'
        
    def params(self):
        return dict([('weight1', self.weight1), ('bias1', self.bias1),
                     ('weight2', self.weight2), ('bias2', self.bias2),
                     ('weight3', self.weight3), ('bias3', self.bias3),
                    ])
    
    def init_params(self, rng):
        rng, self.weight1 = split_and_sample(rng, self.weight1.shape)
        self.weight1 = self.weight1 * (1./jnp.sqrt(self.weight1.shape[1]*self.weight1.shape[2]*self.weight1.shape[3]))
        rng, self.weight2 = split_and_sample(rng, self.weight2.shape)
        self.weight2 = self.weight2 * (1./jnp.sqrt(self.weight2.shape[1]*self.weight2.shape[2]*self.weight2.shape[3]))
        rng, self.weight3 = split_and_sample(rng, self.weight3.shape)
        self.weight3 = self.weight3 * (1./jnp.sqrt(self.weight3.shape[1]*self.weight3.shape[2]*self.weight3.shape[3]))
        return rng, self.params()
    
    def forward(self, p, b, x):
        if len(x.shape) < len(p['weight1'].shape):
            x = jnp.expand_dims(x, 0)
        h = x
        h = lax.conv(h, p['weight1'], (1,1), self.mode) + p['bias1'][None,:,None,None]
        h = F.relu(h) 
        h = lax.conv(h, p['weight2'], (1,1), self.mode) + p['bias2'][None,:,None,None]
        h = F.relu(h)
        h = lax.conv(h, p['weight3'], (1,1), self.mode) + p['bias3'][None,:,None,None]
        h = F.relu(h)
        h = h + x
        return h
    
class LayerNorm(Layer):
    def __init__(self, normalized_dim=-1, name=None):
        super(LayerNorm, self).__init__(name)
        
        self.alpha = jnp.zeros((1))
        self.gamma = jnp.ones((1))
        self.normalized_dim = normalized_dim
        
        if name is None:
            self.name = F'LayerNorm+{rand_string()}'
            
    def params(self):
        return dict([('alpha', self.alpha), ('gamma', self.gamma)])
        
    @partial(jax.jit, static_argnums=(0,))
    def forward(self, p, b, x):
        mu = x.mean(self.normalized_dim, keepdims=True)
        var = x.var(self.normalized_dim, keepdims=True)
        return (x - mu) / (jnp.sqrt(var) + 1e-6) * p['gamma'] + p['alpha']    

class BatchNorm(Layer):
    def __init__(self, dim, coeff=0.95, name=None):
        super(BatchNorm, self).__init__(name)
        
        self.mu = jnp.zeros((dim))
        self.var = jnp.ones((dim))
        self.alpha = jnp.zeros((dim))
        self.gamma = jnp.ones((dim))
        
        self.coeff = coeff
        
        if name is None:
            self.name = F'BatchNorm+{rand_string()}'
            
    def buffers(self):
        return dict({'mu': self.mu, 'var': self.var})
    
    def params(self):
        return dict({'alpha': self.alpha, 'gamma': self.gamma})
    
    @partial(jax.jit, static_argnums=(0,))
    def forward(self, p, b, x):
        x_ = x
        mu = x_.mean(0, keepdims=True)
        var = x_.var(0, keepdims=True)
        x_ = (x_ - mu)/(jnp.sqrt(var) + 1e-6) * p['gamma'] + p['alpha']
        
        new_b = dict({'mu': self.coeff * b['mu'] + (1.-self.coeff) * mu, 
                      'var': self.coeff * b['var'] + (1.-self.coeff) * var})
        
        return x_, new_b
    
    @partial(jax.jit, static_argnums=(0,))
    def forward_eval(self, p, b, x):
        x_ = x
        x_ = (x_ - b['mu'])/(jnp.sqrt(b['var']) + 1e-6) * p['gamma'] + p['alpha']
        
        return x_
    
class BatchNorm2d(Layer):
    def __init__(self, dim, coeff=0.95, name=None):
        super(BatchNorm2d, self).__init__(name)
        
        self.mu = jnp.zeros((dim))
        self.var = jnp.ones((dim))
        self.alpha = jnp.zeros((dim))
        self.gamma = jnp.ones((dim))
        
        self.coeff = coeff
        
        if name is None:
            self.name = F'BatchNorm2d+{rand_string()}'
            
    def buffers(self):
        return dict({'mu': self.mu, 'var': self.var})
    
    def params(self):
        return dict({'alpha': self.alpha, 'gamma': self.gamma})
    
    @partial(jax.jit, static_argnums=(0,))
    def forward(self, p, b, x):
        x_ = jnp.transpose(x, [0, 2, 3, 1]).reshape(-1, x.shape[1])
        mu = x_.mean(0)
        var = x_.var(0)
        x_ = (x_ - mu)/(jnp.sqrt(var) + 1e-6) * p['gamma'] + p['alpha']
        x_ = x_.reshape(x.shape[0], x.shape[2], x.shape[3], -1)
        x_ = jnp.transpose(x_, [0, 3, 1, 2])
        
        new_b = dict({'mu': self.coeff * b['mu'] + (1.-self.coeff) * mu, 
                      'var': self.coeff * b['var'] + (1.-self.coeff) * var})
        
        return x_, new_b
    
    @partial(jax.jit, static_argnums=(0,))
    def forward_eval(self, p, b, x):
        x_ = jnp.transpose(x, [0, 2, 3, 1]).reshape(-1, x.shape[1])
        x_ = (x_ - b['mu'])/(jnp.sqrt(b['var']) + 1e-6) * p['gamma'] + p['alpha']
        x_ = x_.reshape(x.shape[0], x.shape[2], x.shape[3], -1)
        x_ = jnp.transpose(x_, [0, 3, 1, 2])
        
        return x_
    
class MaxPool2d(Layer):
    def __init__(self, kw, kh, name=None):
        super(MaxPool2d, self).__init__(name)
        
        _, self.maxpool = jexp.MaxPool((kw, kh))
        
        if name is None:
            self.name = F'MaxPool2d+{rand_string()}'
        
    @partial(jax.jit, static_argnums=(0,))
    def forward(self, p, b, x):
        return jnp.transpose(self.maxpool(None, jnp.transpose(x, [0, 2, 3, 1])), [0, 3, 1, 2])
    
''' max-pool the set of all feature vectors across the 2-d grid. '''
class SpatialPool2d(Layer):
    def __init__(self, name=None):
        super(SpatialPool2d, self).__init__(name)
        
        if name is None:
            self.name = F'SpatialPool2d+{rand_string()}'
        
    @partial(jax.jit, static_argnums=(0,))
    def forward(self, p, b, x):
        return x.max(-1).max(-1)
    
class Tanh(Layer):
    def __init__(self, name=None):
        super(Tanh, self).__init__(name)
        
        if name is None:
            self.name = F'Tanh+{rand_string()}'
            
    @partial(jax.jit, static_argnums=(0,))
    def forward(self, p, b, x):
        return jnp.tanh(x)
    
class ReLU(Layer):
    def __init__(self, name=None):
        super(ReLU, self).__init__(name)
        
        if name is None:
            self.name = F'ReLU+{rand_string()}'
            
    @partial(jax.jit, static_argnums=(0,))
    def forward(self, p, b, x):
        return F.relu(x)
    
class LeakyReLU(Layer):
    def __init__(self, alpha=0.001, name=None):
        super(LeakyReLU, self).__init__(name)
        
        self.alpha = alpha
        
        if name is None:
            self.name = F'PReLU+{rand_string()}'
        
    @partial(jax.jit, static_argnums=(0,))
    def forward(self, p, b, x):
        return F.leaky_relu(x, alpha=self.alpha)
    
class Softmax(Layer):
    def __init__(self, name=None):
        super(Softmax, self).__init__(name)
        
        if name is None:
            self.name = F'Softmax+{rand_string()}'
            
    @partial(jax.jit, static_argnums=(0,))
    def forward(self, p, b, x):
        x_exp = jnp.exp(x)
        return x_exp / jnp.sum(x_exp)
    
