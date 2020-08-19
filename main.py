from torch_snn.F import update
from torch_snn.neuron import LinearNeuron, InputNeuron, LIFNeuron, PoissonNeuron
from torch_snn.synapse import FixedSynapse, STDPSynapse, ExcitorySynapse, InhibitorySynapse
from torch_snn.monitor import VoltageMonitor, SpikeMonitor, SynapseMonitor, ImageVisualizer
import torch
from dataset import mnist

n1 = PoissonNeuron(28*28)
m1 = SpikeMonitor(n1)
n2 = LIFNeuron(784)
s1 = STDPSynapse(n1, n2, torch.eye(784))
s1.lr = 10
m2 = SpikeMonitor(n2)
s1.w *= torch.ones(784,784)*10
x = 0
for image, label in mnist.train_loader:
    n1.input(image[0][0])
    for j in range(10):
        update()
    x += 1
    if x == 100:
        break



ImageVisualizer(m1, (28, 28),'anim1.mp4')
ImageVisualizer(m2, (28, 28),'anim2.mp4')

