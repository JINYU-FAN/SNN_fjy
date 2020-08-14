import abc
import torch

NEURONS = []
SYNAPSES = []
MONITORS = []

class Neuron(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, n):
        NEURONS.append(self)
        self.size = n
        self.spike = (torch.rand(n) < 0).int() # The neurons are initiated with 50% of probability to spike

    @abc.abstractmethod
    def update(self):
        pass


class Synapse(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, pre, post):
        SYNAPSES.append(self)
        self.pre = pre
        self.post = post

    @abc.abstractmethod
    def update(self):
        pass    



class Monitor(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        MONITORS.append(self)

    @abc.abstractmethod
    def update(self):
        pass   