import abc
import torch

NEURONS = []
SYNAPSES = []

class Neuron(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, n):
        NEURONS.append(self)
        self.size = n
        self.spike = (torch.rand(n) > 0.5).int() # The neurons are initiated with 50% of probability to spike

    @abc.abstractmethod
    def update(self):
        pass


class Synapse(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        SYNAPSES.append(self)

    @abc.abstractmethod
    def update(self):
        pass    