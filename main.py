from torch_snn.F import update
from torch_snn.neuron import LinearNeuron, InputNeuron, LIFNeuron, PoissonNeuron
from torch_snn.synapse import FixedSynapse, STDPSynapse, ExcitorySynapse, InhibitorySynapse
from torch_snn.monitor import VoltageMonitor, SpikeMonitor, SynapseMonitor, ImageVisualizer
import torch
from dataset import mnist

n = PoissonNeuron(28*28)
m = SpikeMonitor(n)
x = 0
for image, label in mnist.train_loader:
    n.input(image[0][0])
    for j in range(10):
        update()
    x += 1
    if x == 100:
        break


ImageVisualizer(m, 'anim.mp4')

