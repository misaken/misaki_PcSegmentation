##########################################################################
## tool.pyでは、inlierに対して再度剛体変換を求めて評価を行っていなかった。
## その部分を修正する。バッチ処理で行う。
## このコードは、tool_use_batch.pyを改良したもの。
##########################################################################

import numpy as np
import pdb, pickle, time, sys

from tools import *

if __name__ == "__main__":
    t_s = time.time()
    pc_array_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A01/A01_pc_array.pkl"

    with open(pc_array_path, "rb") as f:
        pc = pickle.load(f)
    n_frame = pc.shape[0]
    n_points = pc.shape[1]
    print(f"n_frame={n_frame}")
    # 各フレームの差分からクラスタリング
    first_frame = 0
    end_frame = n_frame - 1 # 最終フレームのインデックス
    # end_frame = 200
    skip = 10
    t = 1.0e-3 #誤差範囲
    # d = n_points//20 #誤差範囲t以下の点がd個以上であれば正しい剛体変換
    batch_size = 10000
    n_clusters = 20
    sample_size = 3

    output_dir = f"./result/frames/{first_frame}_{end_frame}_{skip}/"
    darray2video = NDARRAY2VIDEO(pc_array_path, output_dir)
    
    label_history = [] # 各点のラベルの推移
    all_iter = (end_frame - first_frame) // skip
    iter_cnt = 1
    for i in range(first_frame, end_frame-skip+1, skip):
        print(f"*******************************    frame {i} and {i+skip}  ({iter_cnt}/{all_iter})     *****************************")
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
            # 再度求めたinlierをクラスタとして登録
            cluster_labels[zero_idx[refined_inlier_idx]] = label
            print(f"{label}`s n_inlier={refined_inlier_idx.sum()}")
            cluster_inlier = np.append(cluster_inlier, refined_inlier_idx.sum())
            print(f"T=\n{np.round(refined_T, 4)}")
            points_for_genT = np.concatenate([points_for_genT, X[most_fitted_sampled_idx].reshape([1, sample_size, 3])])
            label += 1
    
        zero_idx = np.where(cluster_labels == 0)[0]
        cluster_inlier = np.insert(cluster_inlier, 0, zero_idx.shape[0])
        label_history.append(cluster_labels)
        print("\nRANSAC finished")
        print(f"label unique: {np.unique(cluster_labels)}")
        print(f"labels` inlier num: {cluster_inlier}\n")


        # darray2video = NDARRAY2VIDEO(pc_array_path)
        darray2video.plot_sampled_points(points_for_genT, np.unique(cluster_labels))
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