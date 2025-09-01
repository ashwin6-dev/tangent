from tangent.core import tensor, ops, gradient
import numpy as np

x = tensor.make_variable(np.array([[5, 5, 5]]))
y = tensor.make_variable(np.array([[10, 10, 10]]))

z = ops.add(ops.mul(x, 10), ops.mul(x, y))
partials = gradient.GradientEngine().backward(z)

print(partials[y].item())