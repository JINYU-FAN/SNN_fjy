from .abstract import Neuron
from matplotlib import pyplot as plt
import torch

class InputNeuron(Neuron):
    def __init__(self, n):
        Neuron.__init__(self, n)
        self._step = 0
        self._length = 1
        self.sequence = torch.zeros(1, n)
        
    def input(self, sequence):
        
        self.sequence = sequence

    def update(self):
        self.spike = self.sequence[self._step]
        self._step += 1
        if self._step == self._length:
            self._step = 0
        Neuron.update(self)

class LinearNeuron(Neuron):
    '''
    This is a very idealized neuron model, only for test.
    '''
    def __init__(self, n):
        Neuron.__init__(self, n)
        self.v = torch.zeros(n)
        self.I = 0 # The overall current input for the neuron
        self.Iex = 0 # External current by electrode
    
    def update(self):
        self.I = self.Iex
        self.v += self.I
        if self.v > 60:
            self.spike = 1
            self.v = 0
        else:
            self.spike = 0
        self.I = 0
        Neuron.update(self)

class LIFNeuron(Neuron):
    def __init__(self):
        pass

    def update(self):
        pass