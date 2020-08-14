from .abstract import  Synapse
import torch

class FixedSynapse(Synaspe):
    def __init__(self, pre, post):
        Synapse.__init__(self, pre, post)


    def update(self):
        Synapse.update(self)

class STDPSynapse(Synaspe):
    def __init__(self):
        Synapse.__init__(self)


    def update(self):
        Synapse.update(self)
