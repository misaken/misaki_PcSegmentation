import os, pickle, pdb
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from matplotlib import pyplot as plt
# from tools.tools import *
from tools.tool_clustering import *
from tools.tool_plot import *

labels_path = "result/A04_pc_array/frames/0_188_10/label_history.pkl"
pc_array_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A04/A04_pc_array.pkl"
output_dir = "./result/A04_pc_array/frames/0_188_10/"
with open(labels_path, "rb") as f:
    label_history = pickle.load(f)
with open(pc_array_path, "rb") as f:
    pc = pickle.load(f)

metrics = "l0l2"
k = 12
mykmeans = MyKMeans(metrics=metrics, random_seed=0)
labels = mykmeans.calc_aic(range_k=[10, 13], X=label_history.T, pc=pc[0], labelVec_weight=0.7)