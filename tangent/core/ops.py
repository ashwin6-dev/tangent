from .tensor import Tensor
from .node import Node

import operator

def wrap_constant(c):
    if isinstance(c, Tensor):
        return c
    
    return Tensor(c, constant=True)

class Op(Node):
    @staticmethod
    def apply(self, *args):
        pass

class BinOp(Op):
    op_func = None 

    @classmethod
    def apply(cls, left, right):
        obj = cls()
        obj.left = wrap_constant(left)
        obj.right = wrap_constant(right)
        return obj

    def forward(self):
        if self.op_func is None:
            raise NotImplementedError("op_func must be defined in subclass")
        
        new_tensor = Tensor(self.op_func(self.left.item(), self.right.item()))
        new_tensor.set_result_of(self)
        return new_tensor


class Add(BinOp):
    op_func = operator.add

class Sub(BinOp):
    op_func = operator.sub

class Mul(BinOp):
    op_func = operator.mul

class Pow(BinOp):
    op_func = operator.pow

class MatMul(BinOp):
    @staticmethod
    def op_func(a, b):
        return a @ b