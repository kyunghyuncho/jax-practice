class SGD:
    def __init__(self, model, lr=0.01):
        self.lr = lr
        self.model = model
        
    def step(self, grad):
        for ll in self.model.layers:
            if ll.name not in self.model.params:
                continue
            pp = self.model.params[ll.name]
            gg = grad[ll.name]
            for kk in pp.keys():
                pp[kk] = pp[kk] - self.lr * gg[kk]