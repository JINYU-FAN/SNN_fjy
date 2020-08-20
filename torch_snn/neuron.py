from .abstract import Neuron
import torch


class PoissonNeuron(Neuron):
    dt = 0.001
    def __init__(self, n):
        Neuron.__init__(self, n)
        self.image = torch.rand(self.size)
        self.last_spike_time = torch.ones(n) * 10000

    def input(self, image):
        assert self.image.view(-1, 1).shape == (self.size, 1),"The shape of the image is not compatible."
        self.image = image.view(-1)

    def update(self):
        self.spike = torch.rand(self.size) < self.image
        self.last_spike_time += self.dt
        self.last_spike_time = self.spike * 0 + ~self.spike * self.last_spike_time     
        Neuron.update(self)



class InputNeuron(Neuron):
    dt = 0.001
    def __init__(self, n):
        Neuron.__init__(self, n)
        self._step = 0
        self._length = 1
        self.input([0])
        self.last_spike_time = torch.ones(n) * 10000
        
    def input(self, sequence):
        if isinstance(sequence[0], list):
            assert len(sequence) == self.size, "The input sequence has different size with the neurons."
            self.sequence = torch.Tensor(sequence)
            self._length = len(self.sequence[0])
        else:
            self._length = len(sequence)
            self.sequence = torch.Tensor([sequence]*self.size)

    def update(self):
        self.spike = self.sequence[:,self._step]
        self.last_spike_time += self.dt
        self.last_spike_time = self.spike * 0 + (1-self.spike) * self.last_spike_time        
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
        self._I = 0 # The overall current input for the neuron
        self.Iex = 0 # External current by electrode
    
    def update(self):
        self._I = self.Iex + self.Isyn
        self.v += self._I
        if self.v > 60:
            self.spike = 1
            self.v = 0
        else:
            self.spike = 0
        self._I = 0
        Neuron.update(self)

class LIFNeuron(Neuron):
    dt = 0.001
    Rm = 1e6
    Cm = 3e-8
    Vresting = -0.06
    Vthresh = -0.045
    Vreset = -0.06
    Vinit = -0.06
    Trefract = 3e-3
    Inoise = 0 # 1e-5
    def __init__(self, n):
        Neuron.__init__(self, n)
        self.v = torch.ones(n) * self.Vresting
        self._I = torch.zeros(n)
        self.Iex = torch.randn(n)*0.000001
        self.last_spike_time = torch.ones(n) * 10000

    def update(self):
        self._I = self.Iex + self.Isyn
        self.v += ((self.Vresting - self.v)/self.Rm + self._I) * self.dt / self.Cm  
        self.spike = (self.v > self.Vthresh) & (self.last_spike_time > self.Trefract)
        self.v = self.spike * self.Vreset + ~self.spike * self.v
        self.last_spike_time += self.dt
        self.last_spike_time = self.spike * 0 + ~self.spike * self.last_spike_time
        self._I = torch.zeros(self.size)
        Neuron.update(self)


class IzhikevichNeuron(Neuron):
    Vthresh = 30.
    a = 0.02
    b = 0.2
    c = -55#-65.
    d = 2.
    def __init__(self, n):
        Neuron.__init__(self, n)
        self.v = torch.ones(n) * self.c
        self.u = torch.zeros(n)
        self._I = torch.zeros(n)
        self.Iex = torch.randn(n)*0.000001
        self.last_spike_time = torch.ones(n) * 10000

    def update(self):
        self._I = self.Iex + self.Isyn
        dv = 0.04*self.v**2 + 5.*self.v + 140. - self.u + self._I
        self.v += dv
        du = self.a * (self.b * self.v - self.u)
        self.u += du
        self.spike = (self.v > self.Vthresh)
        self.v = self.spike * self.c + ~self.spike * self.v
        self.u += self.spike * self.d
        Neuron.update(self)



