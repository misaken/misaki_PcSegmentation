##########################################################################
## tool.pyでは、inlierに対して再度剛体変換を求めて評価を行っていなかった。
## その部分を修正する。バッチ処理で行う。
## このコードは、tool_use_batch.pyを改良したもの。
##########################################################################

import numpy as np
import pdb, pickle, time, sys, re
from tools.tool_plot import *
from tools.tool_RigidTransform import *
from tools.tool_etc import *
from tools.tool_clustering import *

if __name__ == "__main__":
    t_s = time.time()
    # pc_array_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A04/A04_pc_array.pkl"
    # pc_array_path = "./data/SAPIEN/8961/SAPIEN_8961_pc_array.pkl"
    pc_array_path = "./data/SAPIEN/101387/SAPIEN_101387_pc_array.pkl"
    # pc_array_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A11/A11_pc_array.pkl"
    data_name = re.search(r'([^/]+)(?=\.pkl$)', pc_array_path).group(1)

    with open(pc_array_path, "rb") as f:
        pc = pickle.load(f)
    n_frames = pc.shape[0]
    n_points = pc.shape[1]
    print(f"n_frames={n_frames}")
    # 各フレームの差分からクラスタリング
    first_frame = 0
    end_frame = n_frames # 最終フレームのインデックス
    skip = 5
    t = 1.0e-3 #誤差範囲
    # d = n_points//20 #誤差範囲t以下の点がd個以上であれば正しい剛体変換
    batch_size = 10000
    n_clusters = 20 # 求める剛体変換の最大個数
    sample_size = 3

    spatial_clustering = False # 剛体変換を求めたあと、DBSCANで空間的に分割するか

    output_dir = f"./result/{data_name}/frames/{first_frame}_{end_frame}_{skip}_{t}{'_withDBSCAN' if spatial_clustering else ''}/"
    
    darray2video = NDARRAY2VIDEO(pc_array_path, output_dir)
    
    label_history = [] # 各点のラベルの推移
    n_all_iter = (end_frame - first_frame) // skip
    iter_cnt = 1
    for i in range(first_frame, end_frame-skip, skip):
        print(f"*******************************    frame {i} and {i+skip}  ({iter_cnt}/{n_all_iter})     *****************************")
        sys.stdout.flush() # nohup.outに書き込むように、明示的にフラッシュ
        X = pc[i]
        # Y = pc[i+skip-1]
        Y = pc[i+skip]
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
            print(f"########    start to calc {j+1}th cluster. cluster_unique={np.unique(cluster_labels)}, label 0 num: {zero_idx.shape[0]}    ########")
            most_fitted_n_inlier = 0
            most_fitted_inlier_idx: bool = None # zero_idxに対するinlierのboolean
            most_fitted_T = None
            n_iter = gen_iter_n(0.99, 0.95)
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
            # inlier内で再度剛体変換を求めてinlier再計算
            refined_T = svd2T(X[zero_idx[most_fitted_inlier_idx]], Y[zero_idx[most_fitted_inlier_idx]])
            refined_Y_pred = rigid_transform(refined_T, X[zero_idx]) #再度求めた剛体変換を、ラベル0全てにかける
            refined_inlier_idx = ((Y[zero_idx] - refined_Y_pred)**2).sum(axis=1) < t
            
            # 各クラスタ内でDBSCANを行う場合はTrue
            # ノイズを-1ラベルに一時的にしておいて、最後にラベル0にする
            if spatial_clustering:
                # 空間的に分かれている部分は別のクラスタに割り振りなおす。
                spatial_labels = dbscan(pc[0][zero_idx[refined_inlier_idx]])
                print(f"spatial label unique: {np.unique(spatial_labels)}")
                if -1 in np.unique(spatial_labels):
                    # いったんノイズの点もラベル振ることにする。
                    # spatial_labels = spatial_labels + 1
                    #ノイズをすべて-1にして、あとから0ラベルにする
                    spatial_labels[spatial_labels==-1] = -label - 1
                cluster_labels[zero_idx[refined_inlier_idx]] = spatial_labels + label
            else:
                # 再度求めたinlierをクラスタとして登録
                cluster_labels[zero_idx[refined_inlier_idx]] = label
            print(f"{label}`s n_inlier={refined_inlier_idx.sum()}")
            cluster_inlier = np.append(cluster_inlier, refined_inlier_idx.sum())
            print(f"T=\n{np.round(refined_T, 4)}")
            points_for_genT = np.concatenate([points_for_genT, X[most_fitted_sampled_idx].reshape([1, sample_size, 3])])
            if spatial_clustering:
                # label += np.unique(spatial_labels).shape[0] # ノイズも別ラベルのとき
                label += int((np.unique(spatial_labels) != -label-1).sum()) # ノイズはすべて0番ラベルのとき
            else:
                label += 1
        
        # DBSCANでノイズの点のラベルを一時的に-1にしていたので、0に戻す。
        if spatial_clustering:
            cluster_labels[cluster_labels==-1] = 0

        zero_idx = np.where(cluster_labels == 0)[0]
        cluster_inlier = np.insert(cluster_inlier, 0, zero_idx.shape[0])
        label_history.append(cluster_labels)
        print("\nRANSAC finished")
        print(f"label unique: {np.unique(cluster_labels)}")
        print(f"labels` inlier num: {cluster_inlier}\n")


        # darray2video = NDARRAY2VIDEO(pc_array_path)
        # darray2video.plot_sampled_points(points_for_genT, np.unique(cluster_labels))
        # darray2video.create(labels=cluster_labels)
        darray2video.create(labels=cluster_labels, frame_idx=[i+skip], create_video=False)

        sys.stdout.flush() # nohup.outに書き込むように、明示的にフラッシュ
        iter_cnt += 1
    with open(output_dir+"label_history.pkl", "wb") as f:
        pickle.dump(np.array(label_history), f)
    t_e = time.time()
    print(f"time : {t_e - t_s}")
    # darray2video = NDARRAY2VIDEO(pc_array_path)
    # darray2video.create()