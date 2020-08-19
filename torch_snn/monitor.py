from .abstract import Monitor, Neuron, Synapse
from matplotlib import pyplot as plt
import torch
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class VoltageMonitor(Monitor):
    def __init__(self, neurons):
        assert isinstance(neurons, Neuron), 'Voltage monitor can only be used to monitor the membrane potential of neurons'
        Monitor.__init__(self)
        self.neurons = neurons
        self.history = []

    def update(self):
        self.history.append(copy.deepcopy(self.neurons.v))
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
        self.history.append(copy.deepcopy(self.neurons.spike))
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

class SynapseMonitor(Monitor):
    def __init__(self, synapses):
        assert isinstance(synapses, Synapse), 'SynapseMonitor can only be used to monitor synapse obejct.'
        Monitor.__init__(self)
        self.synapses = synapses
        self.history = []

    def update(self):
        self.history.append(copy.deepcopy(self.synapses.w))
        Monitor.update(self)

    def plot(self, *args):
        if args == ():
            indexs1 = range(self.synapses.size[0])
            indexs2 = range(self.synapses.size[1])
        else:
            assert len(args) == 2, 'You should use two integer to determine the synapse.'
            indexs1 = (args[0],)
            indexs2 = (args[1],)
        for i in indexs1:
            for j in indexs2:
                history = []
                for w in self.history:
                    history.append(w[i][j])
                plt.plot(history)
        plt.show()


class ImageVisualizer():
    def __init__(self, data, filename):
        
        self.data = data.history
        self.frame = -1
        fig, ax = plt.subplots()
        g = self.generate_data()
        self.mat = ax.matshow(self.generate_data())
        str(1)
        ani = FuncAnimation(fig, self.update, self.data_gen, repeat = False, interval = 100)
        str(1)
        print(ani.save)
        ani.save(filename)

    def generate_data(self):
        return self.data[self.frame].reshape(28,28)

    def update(self, data):
        self.mat.set_data(data)
        return self.mat

    def data_gen(self):
        for i in range(len(self.data)-2):
            self.frame += 1
            yield self.generate_data()