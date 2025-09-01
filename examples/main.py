from tangent.core import tensor, ops
import numpy as np

x = tensor.make_variable(np.array([[5, 5, 5]]))
y = tensor.make_variable(np.array([[10, 10, 10]]))

z = ops.add(ops.pow(x, 3), ops.mul(x, y))
partials = z.backward()

print (partials[x].item())