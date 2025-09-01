from .tensor import *
from .node import Node
import numpy as np
import operator

def wrap_constant(c):
    if isinstance(c, Tensor):
        return c
    
    return Tensor(c, constant=True)

def merge_gradients(grad_map_x, grad_map_y):
    if grad_map_x is None:
        return grad_map_y
    if grad_map_y is None:
        return grad_map_x
    
    merged_grad_map = grad_map_x.copy()

    for key in grad_map_y:
        if key in merged_grad_map:
            merged_grad_map[key] = add(merged_grad_map[key], grad_map_y[key])
        else:
            merged_grad_map[key] = grad_map_y[key]

    return merged_grad_map

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
        
def add(left, right):
    return Add.apply(left, right).forward()

def sub(left, right):
    return Sub.apply(left, right).forward()

def mul(left, right):
    return Mul.apply(left, right).forward()

def pow(left, right):
    return Pow.apply(left, right).forward()

def matmul(left, right):
    return MatMul.apply(left, right).forward()