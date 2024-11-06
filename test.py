import numpy as np

a = np.random.randint(6, size=(10, 3))
idx = [5, 1]

dif = a - a[idx].reshape([2, 1, 3])