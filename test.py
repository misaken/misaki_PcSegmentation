import numpy as np
import pickle, pdb

tmp = np.random.choice(4, size=(10, 6))

with open("tmp4medoids.pkl", "rb") as f:
    data = pickle.load(f)
# pdb.set_trace()
cluster_size = data.shape[0]
dim = data.shape[1]

min_distance = np.inf
test = []
medoid = None
for i in range(cluster_size):
    distance = (data != data[i]).sum()
    test.append(distance)
    if distance < min_distance:
        medoid = data[i]
        min_distance = distance
        print(i)
        print(min_distance)
    

pdb.set_trace()
