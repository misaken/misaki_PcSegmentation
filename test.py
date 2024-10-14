import pickle
import numpy as np

with open("all_labels_0_100.pkl", "rb") as f:
    a = pickle.load(f)

a = np.array(a)
print(a.shape)