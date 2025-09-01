from .node import Node
import numpy as np

class Tensor(Node):
    def __init__(self, value, constant=False):
        self.value = value
        self.constant = constant
        self.result_of = None

    def item(self):
        return self.value
    
    def set_result_of(self, op):
        self.result_of = op

    def backward(self, g=None):
        if self.constant:
            return { self: make_constant(np.zeros_like(self.value)) }
        
        if g is None:
            g = make_constant(np.ones_like(self.value))
        
        if self.result_of is not None:
            partials = self.result_of.backward(g)
            partials[self] = g
            
            return partials

        return { self: g }


def make_variable(value):
    return Tensor(value, constant=False)

def make_constant(value):
    return Tensor(value, constant=True)