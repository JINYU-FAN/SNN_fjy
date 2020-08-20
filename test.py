from torch_snn.F import update
from torch_snn.neuron import LinearNeuron, InputNeuron, LIFNeuron, PoissonNeuron, IzhikevichNeuron
from torch_snn.synapse import FixedSynapse, STDPSynapse, ExcitorySynapse, InhibitorySynapse
from torch_snn.monitor import VoltageMonitor, SpikeMonitor, SynapseMonitor, ImageVisualizer
import torch

n = IzhikevichNeuron(1)
n.Iex = torch.Tensor([20])
m1 = VoltageMonitor(n)
m2 = SpikeMonitor(n)
for i in range(200):
    update()

m1.plot()
m2.plot()
ImageVisualizer(m2, (1,1), 'anim.mp4')
