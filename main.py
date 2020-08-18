from torch_snn.F import update
from torch_snn.neuron import LinearNeuron, InputNeuron, LIFNeuron, PoissonNeuron
from torch_snn.synapse import FixedSynapse, STDPSynapse
from torch_snn.monitor import VoltageMonitor, SpikeMonitor, SynapseMonitor
import torch
import copy
'''
n1 = InputNeuron(2)

n1.input([[0,0,0,0,0],[1,0,1,0,0]])

n2 = InputNeuron(3)
n2.input([[0,1,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
s = STDPSynapse(n1, n2)
s.w = torch.randn(2,3)*10
m1 = SynapseMonitor(s)
m2 = SpikeMonitor(n1)
m3 = SpikeMonitor(n2)

'''
n3 = PoissonNeuron(5)
m4 = SpikeMonitor(n3)
for i in range(100): 
    update()
print(n3.image)
#m2.plot()
#m3.plot()
m4.plot()

