# TEST
import numpy as np

x = np.array(range(10))
print(x)

# multiple inserts at once
indices = np.arange(0, 10, 2)
print(indices)

z = np.insert(x, indices, x[indices])

print(z)