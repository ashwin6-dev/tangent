from tangent.core import tensor, ops
import numpy as np

x = tensor.Tensor(np.array([[5, 5, 5]]))
y = tensor.Tensor(np.array([[2, 2, 2]]))

z = ops.Add.apply(x, y)

print(z.forward().item())