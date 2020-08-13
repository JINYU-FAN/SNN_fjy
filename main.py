from torch_snn.F import update
from torch_snn.neurons import LinearNeuron, InputNeuron


n = InputNeuron(2)
n.input([0,0,1,0,0,0,0,1])
for i in range(20):
    update()
    print(n.spike)