import jax
from jax import numpy as jnp

from utils import apply_dict

class SGD:
    def __init__(self, model, lr=0.01):
        self.lr = lr
        self.model = model
        
    def step(self, grad):
        apply_dict(lambda g, p: p - self.lr * g, grad, self.model.params)
                
class Adam:
    def __init__(self, model, lr=0.001, betas=(0.9, 0.999), eps=1e-6):
        self.model = model
        
        self.lr = lr
        self.betas = betas
        self.eps = eps
        
        self.first_moment = self.init_moment()
        self.second_moment = self.init_moment()
        self.update = self.init_moment()
        
        self.t = 0
        
    def init_moment(self):
        moment = dict()
        
        for ll in self.model.layers:
            if ll.name not in self.model.params:
                continue
            pp = self.model.params[ll.name]
            moment[ll.name] = dict()
            gg = moment[ll.name]
            for kk in pp.keys():
                gg[kk] = jnp.zeros(pp[kk].shape)
                
        return moment

    def step(self, grad):
        self.t = self.t + 1

        apply_dict(lambda g, m: self.betas[0] * m + (1.-self.betas[0]) * g, grad, self.first_moment)
        apply_dict(lambda g, v: self.betas[1] * v + (1.-self.betas[1]) * (g ** 2), grad, self.second_moment)
        
        apply_dict(lambda m, u: m / (1.-(self.betas[0] ** self.t)), self.first_moment, self.update)
        apply_dict(lambda v, u: u / (jnp.sqrt(v / (1.-(self.betas[1] ** self.t))) + self.eps), self.second_moment, self.update)
        
        apply_dict(lambda u, p: p - self.lr * u, self.update, self.model.params)
        