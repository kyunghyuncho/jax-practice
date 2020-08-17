import numpy
import pickle

from jax import random as jrng
from jax import numpy as jnp
import jax

from functools import partial

from utils import rand_string, split_and_sample, apply_dict

class Model:
    def __init__(self, rng, layers, loss=None, name=None, devices=None):
        if name is None:
            name = F'Model+{rand_string()}'
            
        if devices is None:
            devices = jax.devices()
        self.devices = devices
            
        self.layers = layers
        
        if type(loss) is not list:
            self.loss = [(loss, 1.)]
        else:
            self.loss = loss
            
        self.loss_grad_ = jax.grad(self.loss_, has_aux=True)
        self.loss_grad_eval_ = jax.grad(self.loss_eval_)
        
        self._init_params(rng)
        
        self.eval_ = False
        
    def train(self):
        self.eval_ = False
        
    def eval(self):
        self.eval_ = True
        
    def _init_params(self, rng=None):
        self.params = dict()
        self.buffers = dict()

        for ll in self.layers:
            if rng is None:
                pp = ll.params()
            else:
                rng, pp = ll.init_params(rng)
            if pp is not None:
                self.params[ll.name] = pp
            buffer = ll.buffers()
            if buffer is not None:
                self.buffers[ll.name] = buffer                
        
    def save_state(self, f):
        with open(f, 'wb') as fobj:
            pickle.dump([ll.name for ll in self.layers], fobj, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.params, fobj, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.buffers, fobj, protocol=pickle.HIGHEST_PROTOCOL)
            
    def load_state(self, f):
        with open(f, 'rb') as fobj:
            layer_names = pickle.load(fobj)
            
            if len(self.layers) != len(layer_names):
                print('layers do not match: saved {} current {}'.format(
                layer_names, [ll.name for ll in self.layers]))
                return
            
            for ll, name in zip(self.layers, layer_names):
                ll.name = name
                
            self._init_params()
        
            pload = pickle.load(fobj)
            apply_dict(lambda f, t: f, pload, self.params)
            
            bload = pickle.load(fobj)
            apply_dict(lambda f, t: f, bload, self.buffers)

    @partial(jax.jit, static_argnums=(0,))
    def forward_(self, p, b, x):
        new_b = dict()
        h = x
        for ll in self.layers:
            out = ll(None if ll.name not in p else p[ll.name], 
                   None if ll.name not in b else b[ll.name],
                   h)
            if type(out) == tuple:
                new_b[ll.name] = out[1]
                h = out[0]
            else:
                h = out
                
        return h, new_b
    
    @partial(jax.jit, static_argnums=(0,))
    def forward_eval_(self, p, b, x):
        h = x
        for ll in self.layers:
            h = ll.forward_eval(None if ll.name not in p else p[ll.name], 
                                None if ll.name not in b else b[ll.name],
                                h)
        return h    

    @partial(jax.jit, static_argnums=(0,))
    def loss_(self, p, b, x, y):
        def dummy(mymodel, params, buffers, x, y):
            total_l = 0.
            for ll in mymodel.loss:
                total_l = total_l + ll[1] * ll[0](params, buffers, x, y)
            return total_l
        out, new_buffer = self.forward_(p, b, x)
        return jax.vmap(dummy, in_axes=(None,None,None,0,0))(self, self.params, self.buffers, out, y).mean(), new_buffer

    @partial(jax.jit, static_argnums=(0,))
    def loss_eval_(self, p, x, y):
        def dummy(mymodel, params, buffers, x, y):
            total_l = 0.
            for ll in mymodel.loss:
                total_l = total_l + ll[1] * ll[0](params, buffers, x, y)
            return total_l
        return jax.vmap(dummy, in_axes=(None,None,None,0,0))(self, self.params, self.buffers, self.forward_eval(p, b, x), y).mean()
    
    def forward(self, x, single=False):
        if self.eval_:
            if single:
                return self.forward_eval_(self.params, self.buffers, x)

            def dummy(mymodel, params, buffers, x):
                return mymodel.forward_eval_(params, buffers, x)
            return jax.vmap(dummy, in_axes=(None, None, None, 0))(self, self.params, self.buffers, x)
        else:
            if single:
                return self.forward_(self.params, self.buffers, x)

            def dummy(mymodel, params, buffers, x):
                return mymodel.forward_(params, buffers, x)
            return jax.vmap(dummy, in_axes=(None, None, None, 0))(self, self.params, self.buffers, x)
    
    def loss_grad(self, x, y):
        if self.eval_:
            return self.loss_eval_(self.params, self.buffers, x, y), self.loss_grad_eval_(self.params, self.buffers, x, y)
        else:
            loss, buffer = self.loss_(self.params, self.buffers, x, y)
            grad = self.loss_grad_(self.params, self.buffers, x, y)[0]
            
            apply_dict(lambda g, p: g, buffer, self.buffers)
            
            return loss, grad
    
    def _perturb_dict(self, dict_, rng, scale=1.):
        for kk, vv in dict_.items():
            if type(vv) == dict:
                rng = self._perturb_dict(vv, rng, scale=scale)
            else:
                rng, noise = split_and_sample(rng, vv.shape)
                dict_[kk] = vv + scale * noise
        return rng
    
    def perturb(self, rng, scale=1.):
        return self._perturb_dict(self.params, rng, scale=scale)
        
