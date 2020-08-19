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
        self.Isyn = torch.zeros(n)

    @abc.abstractmethod
    def update(self):
        pass


class Synapse(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, pre, post):
        SYNAPSES.append(self)
        self.pre = pre
        self.post = post
        self.size = (self.pre.size, self.post.size)
        self.w = torch.zeros(self.pre.size, self.post.size)

    @abc.abstractmethod
    def update(self):
        self.post.Isyn = torch.mm(torch.unsqueeze(self.pre.spike.float(), 0), self.w).squeeze()


class Monitor(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        MONITORS.append(self)

    @abc.abstractmethod
    def update(self):
        pass   