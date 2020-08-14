from torch_snn.F import update
from torch_snn.neuron import LinearNeuron, InputNeuron, LIFNeuron
from torch_snn.monitor import VoltageMonitor, SpikeMonitor


n = LIFNeuron(3)
m1 = SpikeMonitor(n)
m2 = VoltageMonitor(n)
n._I = 0.000000001
for i in range(100):
    update()

m1.plot()
m2.plot()
