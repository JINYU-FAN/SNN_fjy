from .abstract import Monitor, Neuron, Synapse
from matplotlib import pyplot as plt
import torch

class VoltageMonitor(Monitor):
    def __init__(self, neurons):
        assert isinstance(neurons, Neuron), 'Voltage monitor can only be used to monitor the membrane potential of neurons'
        Monitor.__init__(self)
        self.neurons = neurons
        self.history = []

    def update(self):
        self.history.append(self.neurons.v)
        Monitor.update(self)

    def plot(self, index = None):
        if index == None:
            indexs = range(self.neurons.size)
        else:
            indexs = (index,)
        for index in indexs:
            history = []
            for v in self.history:
                history.append(v[index])
            plt.plot(history)
        plt.show()
        

class SpikeMonitor(Monitor):
    def __init__(self, neurons):
        assert isinstance(neurons, Neuron), 'Voltage monitor can only be used to monitor the membrane potential of neurons'
        Monitor.__init__(self)
        self.neurons = neurons
        self.history = []

    def update(self):
        self.history.append(self.neurons.spike)
        Monitor.update(self)

    def plot(self, index=None):
        if index == None:
            indexs = range(self.neurons.size)
        else:
            indexs = (index,)
        for index in indexs:
            steps = []
            history = []
            for step, spike in enumerate(self.history):
                if spike[index] == 1:
                    steps.append(step)
                    history.append(int(spike[index])*index)
            plt.scatter(steps, history)
        plt.show()

