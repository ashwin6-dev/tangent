import numpy as np
from .tensor import Tensor, make_constant
from .ops import Add, Sub, Mul, Pow, MatMul, add, sub, mul, pow, matmul

def _grad_add(op, out_grad):
    return [(op.left, out_grad), (op.right, out_grad)]

def _grad_sub(op, out_grad):
    return [(op.left, out_grad), (op.right, mul(out_grad, make_constant(-1)))]

def _grad_mul(op, out_grad):
    return [
        (op.left, mul(out_grad, op.right)),
        (op.right, mul(out_grad, op.left))
    ]

def _grad_pow(op, out_grad):
    x = op.left
    y = op.right
    left_grad = mul(mul(y, pow(x, sub(y, make_constant(1)))), out_grad)
    log_x = make_constant(np.log(x.item())) if np.all(x.item() > 0) else make_constant(0)
    right_grad = mul(mul(pow(x, y), log_x), out_grad)
    return [
        (x, left_grad),
        (y, right_grad)
    ]

def _grad_matmul(op, out_grad):
    left = op.left
    right = op.right
    left_grad = matmul(out_grad, make_constant(right.item().T))
    right_grad = matmul(make_constant(left.item().T), out_grad)
    return [
        (left, left_grad),
        (right, right_grad)
    ]

_GRADIENT_REGISTRY = {
    Add: _grad_add,
    Sub: _grad_sub,
    Mul: _grad_mul,
    Pow: _grad_pow,
    MatMul: _grad_matmul,
}

class GradientEngine:
    def __init__(self, registry=None):
        self.registry = registry or _GRADIENT_REGISTRY

    def backward(self, output_tensor, grad=None):
        if grad is None:
            grad = make_constant(np.ones_like(output_tensor.item()))
        grads = {}
        self._backward(output_tensor, grad, grads)
        return grads

    def _backward(self, tensor, grad, grads):
        if tensor.constant:
            return
        
        if tensor not in grads:
            grads[tensor] = make_constant(np.zeros_like(tensor.item()))
            
        grads[tensor] = add(grads[tensor], grad)
        op = getattr(tensor, 'result_of', None)
        if op is not None:
            grad_fn = self.registry.get(type(op))
            if grad_fn is not None:
                for parent, parent_grad in grad_fn(op, grad):
                    self._backward(parent, parent_grad, grads)
