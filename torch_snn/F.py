import time
from .abstract import NEURONS, SYNAPSES

def update():
    start_time = time.time()
    for n in NEURONS:
        n.update()
    for s in SYNAPSES:
        s.update()

    print(f"STEP TIME:{time.time() - start_time:.09f}s")
