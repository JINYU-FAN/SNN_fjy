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
    lr = 1 # learning rate
    def __init__(self, pre, post, connection = None):
        Synapse.__init__(self, pre, post)
        if connection == None:
            connection = torch.ones(self.size)
        self.connection = connection

    @property
    def connection(self):
        return self._connection

    @connection.setter
    def connection(self, connection):
        self._connection = connection
        self.w *= self._connection


    def update(self):
        Synapse.update(self)
        dw = self.Ap * torch.exp(-self.pre.last_spike_time/self.taup)
        self.w += torch.mm(dw.unsqueeze(1), self.post.spike.unsqueeze(0).float()) * self.lr
        dw = self.Am * torch.exp(-self.post.last_spike_time/self.taum)
        self.w -= torch.mm(self.pre.spike.unsqueeze(1).float(), dw.unsqueeze(0)) * self.lr
        self.w *= self._connection

    def randomize(self, min, max):
        Synapse.randomize(self, min, max)
        self.w *= self._connection

class ExcitorySynapse(STDPSynapse):
    def __init__(self, pre, post, connection = None):
        STDPSynapse.__init__(self, pre, post, connection)
        self.w = torch.clamp(self.w, min = 0)


    def update(self):
        STDPSynapse.update(self)
        self.w = torch.clamp(self.w, min = 0)

    def randomize(self, min, max):
        STDPSynapse.randomize(self, min, max)
        self.w = torch.clamp(self.w, min = 0)


class InhibitorySynapse(STDPSynapse):
    '''
    I am still not sure about the learning rule of inhibitory neuron.
    '''
    Ap = -1
    Am = -1
    def __init__(self, pre, post, connection = None):
        STDPSynapse.__init__(self, pre, post, connection)
        self.w = torch.clamp(self.w, max = 0)

    def update(self):
        STDPSynapse.update(self)
        self.w = torch.clamp(self.w, max = 0)

    def randomize(self, min, max):
        STDPSynapse.randomize(self, min, max)
        self.w = torch.clamp(self.w, max = 0)