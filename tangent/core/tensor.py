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

def make_variable(value):
    return Tensor(value, constant=False)

def make_constant(value):
    return Tensor(value, constant=True)