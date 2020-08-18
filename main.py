from torch_snn.F import update
from torch_snn.neuron import LinearNeuron, InputNeuron, LIFNeuron, PoissonNeuron
from torch_snn.synapse import FixedSynapse, STDPSynapse, ExcitorySynapse, InhibitorySynapse
from torch_snn.monitor import VoltageMonitor, SpikeMonitor, SynapseMonitor
import torch
from dataset import mnist

for image, label in mnist.train_loader:
    print(label)

