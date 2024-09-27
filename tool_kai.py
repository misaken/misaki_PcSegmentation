##########################################################################
## tool.pyでは、inlierに対して再度剛体変換を求めて評価を行っていなかった。
## その部分を修正する。バッチ処理で行う。
## このコードは、tool_use_batch.pyを改良したもの。
##########################################################################

import numpy as np
from numpy.linalg import svd, matrix_rank
import open3d as o3d
import pdb, pickle, time, torch

from tools import *

# #pyplotの座標の表現が行ベクトルで各点を表しているので、列ベクトルに修正した後変換したものをもとの形で返す。
# def rigid_transform_batch(T, X):
#     batch_size = T.shape[0]
#     # Tは同次座標, Xはpyplotで描画する前提の点集合。
#     num_data = X.shape[0]
#     X_T = X.T
#     ones_vector = np.ones([1, num_data])
#     X_ex = np.concatenate([X_T, ones_vector], 0)
#     X_ex = X_ex.reshape([1, 4, num_data]).repeat(batch_size, axis=0)
#     transformed = (T @ X_ex)[:, :-1, :].transpose(0, 2, 1)
#     return transformed

# def svd2T_batch(X, Y):
#     # pdb.set_trace()
#     X = X.transpose(0, 2, 1)
#     Y = Y.transpose(0, 2, 1)
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#     X = torch.tensor(X).to(device)
#     Y = torch.tensor(Y).to(device)
#     num_data = X.shape[2]
#     batch_size = X.shape[0]
#     # X_centroid = (np.sum(X, axis=1) / num_data).reshape(3,1)
#     # Y_centroid = (np.sum(Y, axis=1) / num_data).reshape(3,1)
#     X_centroid = (torch.sum(X, dim=2) / num_data).reshape(batch_size, 3, 1)
#     Y_centroid = (torch.sum(Y, dim=2) / num_data).reshape(batch_size, 3, 1)
#     X_, Y_ = X - X_centroid, Y - Y_centroid
#     # A = X_ @ Y_.T
#     A = torch.matmul(X_, Y_.transpose(2, 1))
#     A = A.to(device)
#     #u, s, v = svd(A)
#     u, s, vT = torch.linalg.svd(A, full_matrices=True)
#     # pdb.set_trace()
    
#     # R = v.T @ u.T
#     R = torch.bmm(vT.transpose(1, 2), u.transpose(1, 2))
#     detR = torch.linalg.det(R)
#     #print(f"det(R) = {detR}  s={s} vT={vT}")
#     #回転行列の行列式が-1の場合は反射になっているので、その時の対処
#     #特異値の一つが0の場合は、その対応するvの符号を反転する
#     detR_int = torch.round(detR).to(torch.int)
#     reflect_index = torch.where(detR_int == -1)[0]
#     for i in reflect_index:
#         if sum(s[i] <= 1.0e-10) != 1:
#             print("can not find R")
#         else:
#             sig_change_index = torch.where(s[i] <= 1.0e-10)
#             vT[sig_change_index] = vT[sig_change_index] * -1
#             R = vT.T @ u.T
#     t = (Y_centroid - R @ X_centroid).reshape(batch_size, 3)
#     # t = (Y_centroid - R @ X_centroid)

#     T = torch.stack([torch.eye(4) for _ in range(batch_size)])
#     T[:, :3, :3] = R
#     T[:, :3, 3] = t

#     return T.numpy()



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
    #ply2ndarray("./data/pointclouds/bodyHands_REGISTRATIONS_A01/", "./A01_pc_array.pkl")
    with open("./A01_pc_array.pkl", "rb") as f:
        pc = pickle.load(f)
    n_frame = pc.shape[0]
    n_points = pc.shape[1]
    print(f"n_frame={n_frame}")

    # 各フレームの差分からクラスタリング
    first_frame = 0
    n_frame = 20
    skip = 20
    t = 1.0e-3 #誤差範囲
    # d = n_points//20 #誤差範囲t以下の点がd個以上であれば正しい剛体変換
    batch_size = 10000
    n_clusters = 3
    sample_size = 3
    for i in range(first_frame, first_frame+n_frame+1-skip, skip):
    # for i in range(n_frame-1):
        X = pc[i]
        Y = pc[i+skip-1]
        # n_iter = 10000
        n_iter = gen_iter_n(0.99, 0.95)
        points_for_genT = np.empty([0, sample_size, 3], dtype=int) # inlierの個数がしきい値以上の剛体変換を求めるときに使った点
        # 各点が属するクラスターラベル初期化.0がクラス無し
        cluster_labels = np.zeros([n_points,], dtype=int)
        label=1
        cluster_inlier = np.array([])
        for j in range(n_clusters): #クラスター数だけ繰り返す。
            zero_idx = np.where(cluster_labels == 0)[0]
            if zero_idx.shape[0] < sample_size:
                print("The number of label 0 is smaller than sample size.")
                break
            print(f"########    start to calc {j+1}th cluster. cluster_unique={np.unique(cluster_labels)}, label 0 num: {zero_idx.shape[0]}")
            most_fitted_n_inlier = 0
            most_fitted_inlier_idx: bool = None # zero_idxに対するinlierのboolean
            most_fitted_T = None
            t_s = time.time()
            for k in range(n_iter // batch_size+1):
                print(f"iteration:  {k+1} / {n_iter // batch_size+1}")
                # クラス0の点からサンプリング
                # sampled_idx = np.random.choice(zero_idx, sample_size, replace=False)
                sampled_idx = sampling_by_distance_batch(X, zero_idx, batch_size, sample_size)
                # sampled_idx = np.array([np.random.choice(zero_idx, sample_size, replace=False) for _ in range(batch_size)])
                # pdb.set_trace()
                T = svd2T_batch(X[sampled_idx], Y[sampled_idx]) # T: (batch_size, 4, 4)
                #print(T)
                Y_pred = rigid_transform_batch(T, X[zero_idx]) #Y_pred : (batch_size, num_data, 3)
                norm = ((Y[zero_idx] - Y_pred)**2).sum(axis=2)
                # np.append(norm_list, norm)
                inlier_idx = norm < t
                n_inlier = inlier_idx.sum(axis=1)
                max_inlier_idx = n_inlier.argmax()
                if n_inlier[max_inlier_idx] > most_fitted_n_inlier:
                    most_fitted_n_inlier = n_inlier[max_inlier_idx]
                    most_fitted_T = T[max_inlier_idx]
                    most_fitted_inlier_idx = inlier_idx[max_inlier_idx]
                    most_fitted_sampled_idx = sampled_idx[max_inlier_idx]
                    print(f"most fitted inlier T`s num: {most_fitted_n_inlier}")
            # print(f"avg = {norm_list.mean()}\ns^2 = {sum((norm_list - norm_list.mean())**2) / norm_list.shape[0]}")
            t_e = time.time()
            print(f"time : {t_e - t_s}")
            # inlier内で再度剛体変換を求めてinlier再計算
            refined_T = svd2T(X[zero_idx[most_fitted_inlier_idx]], Y[zero_idx[most_fitted_inlier_idx]])
            refined_Y_pred = rigid_transform(refined_T, X[zero_idx]) #再度求めた剛体変換を、ラベル0全てにかける
            refined_inlier_idx = ((Y[zero_idx] - refined_Y_pred)**2).sum(axis=1) < t
            # 再度求めたinlierをクラスタとして登録
            cluster_labels[zero_idx[refined_inlier_idx]] = label
            print(f"{label}`s n_inlier={refined_inlier_idx.sum()}")
            cluster_inlier = np.append(cluster_inlier, refined_inlier_idx.sum())
            print(f"T=\n{np.round(most_fitted_T, 4)}")
            points_for_genT = np.concatenate([points_for_genT, X[most_fitted_sampled_idx].reshape([1, sample_size, 3])])
            label += 1
    
    zero_idx = np.where(cluster_labels == 0)[0]
    cluster_inlier = np.insert(cluster_inlier, 0, zero_idx.shape[0])
    print("\nRANSAC finished")
    print(f"label unique: {np.unique(cluster_labels)}")
    print(f"labels` inlier num: {cluster_inlier}\n")
    
    
    darray2video = NDARRAY2VIDEO("A01_pc_array.pkl")
    darray2video.plot_sampled_points(points_for_genT, np.unique(cluster_labels))
    darray2video.create(labels=cluster_labels)

    
    # darray2video = NDARRAY2VIDEO("A01_pc_array.pkl")
    # darray2video.create()