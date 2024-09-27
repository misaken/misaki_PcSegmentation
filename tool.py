# import numpy as np
# from numpy.linalg import svd, matrix_rank
# import open3d as o3d
# import subprocess
# import pdb, pickle, os, cv2, time
# from matplotlib import pyplot as plt
# from sklearn.cluster import KMeans
# from scipy.spatial.distance import mahalanobis
# from scipy.spatial.distance import cdist
# from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy.linalg import svd, matrix_rank
import open3d as o3d
import pdb, pickle, time, torch

from tools import *


if __name__ == "__main__":
    #create video from point clouds
    # ndarray2video = NDARRAY2VIDEO()
    # ndarray2video.create(create_frame_flg=True)


    # # dir_path = "./support_data/my_pointcloud/bodyHands_REGISTRATIONS_A01/"
    # # file_n = int(subprocess.run(f"ls -1 {dir_path} | wc -l", shell=True, capture_output=True, text=True).stdout) - 1
    # # points_array = []

    # # for i in range(file_n):
    # #     file_path = dir_path + str(i).zfill(4) + ".ply"
    # #     point_cloud = o3d.io.read_point_cloud(file_path)
    # #     pdb.set_trace()
    # #     points = np.asarray(point_cloud.points)
    # #     points_array.append(points)
    # # points_array = np.stack(points_array)
    
    
    # #create_colors()
    # with open("color_labels.pkl", "rb") as f:
    #     colors = pickle.load(f)
    # with open("cluster_labels.pkl", "rb") as f:
    #     clusters = pickle.load(f)
    # with open("points_array.pkl", "rb") as f:
    #     points = pickle.load(f)
    # pdb.set_trace()


    #--------------------------------------------------------------------------------------
    with open("./A01_pc_array.pkl", "rb") as f:
        pc = pickle.load(f)
    n_frame = pc.shape[0]
    n_points = pc.shape[1]
    print(f"n_frame={n_frame}")

    # 各フレームの差分からクラスタリング
    n_frame = 20
    skip = 20
    sample_size = 3
    n_iter = 10000
    t = 1.0e-3 #誤差範囲
    d = n_points//40 #誤差範囲t以下の点がd個以上であれば正しい剛体変換

    for i in range(0, n_frame+1-skip, skip):
        X = pc[i]
        Y = pc[i+skip]
        n_clusters = 5
        points_for_genT = np.empty([0, sample_size, 3], dtype=int) # inlierの個数がしきい値以上の剛体変換を求めるときに使った点
        # 各点が属するクラスターラベル.0がクラス無し
        cluster_labels = np.zeros([n_points,], dtype=int)
        label=1
        for j in range(n_clusters): #クラスター数だけ繰り返す。
            # pdb.set_trace()
            print(f"start {j}th iteration. cluster_unique={np.unique(cluster_labels)}")
            zero_idx = np.where(cluster_labels == 0)[0]
            print(f"zero_idx.shape={zero_idx.shape}")
            if zero_idx.shape[0] < 3:
                break
            t_s = time.time()
            for k in range(n_iter):
                # クラス0の点からサンプリング
                # random_idx = np.random.choice(zero_idx, sample_size, replace=False)
                # 基準点からユークリッド距離がしきい値以下のものから選ぶようにする。
                random_idx = sampling_by_distance(X, zero_idx, sample_size)
                # random_idx = np.array([1000, 1001, 1002])
                T = svd2T(X[random_idx, :], Y[random_idx, :])
                #print(T)
                Y_pred = rigid_transform(T, X)
                norm = ((Y - Y_pred)**2).sum(axis=1)
                inlier_idx = norm < t
                n_inlier = sum(inlier_idx)
                if n_inlier > d:
                    #クラスが未定(0)のうち、inlierであったものを新しくラベル付けする
                    cluster_labels[inlier_idx * (cluster_labels==0)] = label
                    print(f"n_inlier={n_inlier}, d={d}")
                    print(f"label={label},  iter={k}")
                    print(f"T=\n{np.round(T, 4)}")
                    points_for_genT = np.concatenate([points_for_genT, X[random_idx].reshape([1, sample_size, 3])])
                    label += 1
                    break
            else:
                print("can not fix")
            t_e = time.time()
            print(f"time : {t_e - t_s}")

        print(np.unique(cluster_labels))
    print(f"sampled points=\n{points_for_genT}")
    darray2video = NDARRAY2VIDEO("A01_pc_array.pkl")
    darray2video.plot_sampled_points(points_for_genT, np.unique(cluster_labels))
    darray2video.create(labels=cluster_labels)

    
    # darray2video = NDARRAY2VIDEO("A01_pc_array.pkl")
    # darray2video.create()