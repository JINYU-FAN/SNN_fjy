from torch_snn.F import update
from torch_snn.neurons import LinearNeuron, InputNeuron


n = InputNeuron(10)
n.input([0,0,1,0,0,0,0,1])
for i in range(5):
    update()