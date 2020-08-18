from .abstract import Synapse
import torch

class FixedSynapse(Synapse):
    def __init__(self, pre, post):
        Synapse.__init__(self, pre, post)


    def update(self):
        
        Synapse.update(self)

class STDPSynapse(Synapse):
    Ap = 1 # A+
    Am = 1 # A-
    taup = 0.001 # second
    taum = 0.0015 # second
    lr = 0.01 # learning rate
    def __init__(self, pre, post):
        Synapse.__init__(self, pre, post)

    def update(self):
        Synapse.update(self)


        dw = self.Ap * torch.exp(-self.pre.last_spike_time/self.taup)
        self.w += torch.mm(dw.unsqueeze(1), self.post.spike.unsqueeze(0).float()) * self.lr
        dw = self.Am * torch.exp(-self.post.last_spike_time/self.taum)
        self.w -= torch.mm(self.pre.spike.unsqueeze(1).float(), dw.unsqueeze(0)) * self.lr

