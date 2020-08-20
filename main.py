from torch_snn.F import update
from torch_snn.neuron import LinearNeuron, InputNeuron, LIFNeuron, PoissonNeuron
from torch_snn.synapse import FixedSynapse, STDPSynapse, ExcitorySynapse, InhibitorySynapse
from torch_snn.monitor import VoltageMonitor, SpikeMonitor, SynapseMonitor, ImageVisualizer
import torch
from dataset import mnist

n1 = PoissonNeuron(28*28)
n2 = LIFNeuron(100)
n3 = LIFNeuron(100)
m1 = SpikeMonitor(n2)
m2 = VoltageMonitor(n2)
s1 = STDPSynapse(n1,n2)
s1.randomize(0.3,0.7)
s2 = ExcitorySynapse(n2, n3, connection=torch.eye(100))
s2.randomize(10, 50)
#s3 = InhibitorySynapse(n3, n2, connection=1-torch.eye(100))
#s3.randomize(-50,-50)

x = 0
for image, label in mnist.train_loader:
    n1.input(image[0][0])
    for j in range(10):
        update()
    x += 1
    if x == 10:
        break


m1.plot()
ImageVisualizer(m1, (10, 10),'anim1.mp4')
ImageVisualizer(m2, (10, 10),'anim2.mp4')

