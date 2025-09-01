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

    def backward(self, g=None):
        if g is None:
            g = make_constant(np.ones_like(self.left.item()))

        return merge_gradients(self.left.backward(g), self.right.backward(g))

class Sub(BinOp):
    op_func = operator.sub

    def backward(self, g=None):
        if g is None:
            g = make_constant(np.ones_like(self.left.item()))

        right_g = mul(g, make_constant(-1))

        return merge_gradients(self.left.backward(g), self.right.backward(right_g))

class Mul(BinOp):
    op_func = operator.mul

    def backward(self, g=None):
        if g is None:
            g = make_constant(np.ones_like(self.left.item()))

        left_g = mul(g, self.right)
        right_g = mul(g, self.left)

        return merge_gradients(self.left.backward(left_g), self.right.backward(right_g))

class Pow(BinOp):
    op_func = operator.pow

    def backward(self, g=None):
        if g is None:
            g = make_constant(np.ones_like(self.left.item()))

        x = self.left
        y = self.right
        left_g = mul(mul(y, pow(x, sub(y, make_constant(1)))), g)

        log_x = make_constant(np.log(x.item())) if np.all(x.item() > 0) else make_constant(0)
        right_g = mul(mul(pow(x, y), log_x), g)

        return merge_gradients(x.backward(left_g), y.backward(right_g))

class MatMul(BinOp):
    def backward(self, g=None):
        if g is None:
            g = make_constant(np.ones_like(self.left.item() @ self.right.item()))

        left_g = matmul(g, make_constant(self.right.item().T))
        right_g = matmul(make_constant(self.left.item().T), g)

        return merge_gradients(self.left.backward(left_g), self.right.backward(right_g))
    
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