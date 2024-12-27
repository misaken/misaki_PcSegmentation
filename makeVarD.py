##########################################################################
## 距離の分散ベースでのセグメンテーション。
## 同じ剛体に属する二点間の距離は変化しない。
## これを利用し、全フレームでの距離の分散が小さい二点は同じ剛体とする。
## このとき、距離の平均値がしきい値以上の二点に関しては別剛体とする。
###########################################################################

import numpy as np
import pdb, pickle, time, sys, re
from itertools import combinations

from tools.tool_plot import *
# from tools.tool_RigidTransform import *
# from tools.tool_etc import *
# from tools.tool_clustering import *

if __name__ == "__main__":
    t_s = time.time()
    pc_array_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A04/A04_pc_array.pkl"
    data_name = re.search(r'([^/]+)(?=\.pkl$)', pc_array_path).group(1)

    with open(pc_array_path, "rb") as f:
        pc = pickle.load(f)
    
    n_frames = pc.shape[0]
    n_points = pc.shape[1]
    # n_points = 4
    
    first_frame = 0
    end_frame = n_frames -1 # 最終フレームのインデックス
    skip = 10

    output_dir = f"./result/{data_name}/DVariance_allPoints/{first_frame}_{end_frame}_{skip}/"
    
    darray2video = NDARRAY2VIDEO(pc_array_path, output_dir)
    
    
    sum_distances_squared = np.zeros([n_points, n_points])
    sum_distances = np.zeros([n_points, n_points])

    n_all_iter = ((end_frame - first_frame) // skip) + 1
    for i in range(first_frame, end_frame+1, skip):
        print(i)
        X = pc[i]
        # X = np.random.choice(range(5), size=[4, 3])
        n_pairs = n_points * (n_points-1) // 2

        distances_squared = (((X.reshape(n_points, 1, 3) - X.reshape(1, n_points, 3)).reshape([n_points, n_points, 3]))**2).sum(axis=2)
        distances = np.sqrt(distances_squared)
        sum_distances_squared += distances_squared
        sum_distances += distances
    
    avg_distances_squared = sum_distances_squared / n_all_iter
    avg_distances = sum_distances / n_all_iter
    var_distances = avg_distances_squared - avg_distances**2
    
    with open(output_dir+"avg_distances.pkl", "wb") as f:
        pickle.dump(avg_distances, f)
    with open(output_dir+"var_distances.pkl", "wb") as f:
        pickle.dump(var_distances, f)
    
    
        # pairs = combinations(range(n_points), 2)
        # for p in pairs:
        #     pdb.set_trace()
        #     a = (X[p[0]] - X[p[1]])**2

        # X_ = X.reshape([n_points, 1, 3])
        # Y_ = Y.reshape([1, n_points, 3])
        # pdb.set_trace()

        # #distancese(n, m): nがiフレームの
        # distances = (((X_ - Y_).reshape([n_points*n_points, 3]))**2).sum(axis=1).reshape(n_points, n_points)
