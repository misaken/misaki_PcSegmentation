import os, pickle, pdb
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from matplotlib import pyplot as plt
# from tools.tools import *
from tools.tool_clustering import *
from tools.tool_plot import *

def whitening(X: np.ndarray):
    """ whitening
    Args:
        X(ndarray): (次元数, データサイズ)の行列
    Return:
        ndarray: 入力Xをwhiteningしたもの
    """
    dim = X.shape[0]
    n = X.shape[1]
    mu = np.mean(X, axis=1).reshape((dim, 1))
    C = ((X - mu)@(X - mu).T) / n
    D, P = np.linalg.eig(C)
    eps = 1e-10
    D = np.clip(D, eps, None)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    X_white = D_inv_sqrt@P.T@(X-mu)
    
    return X_white

def normalize(X):
    """各点のベクトルの大きさが1になるように正規化"""
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    X_normalized = X / norms
    return X_normalized

def k_means(X, k):
    X = X.T
    n_clusters = k
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(X)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    print("クラスタラベル:", labels)
    print("クラスタの重心:\n", centroids)
    # pdb.set_trace()
    return labels

def k_means_custom_d(X, k, metrics="l0"):
    X = X.T # (点の数, ラベル遷移ベクトルのサイズ)の行列になるように転置
    # 距離行列の計算
    if metrics == "l0":
        n_data = X.shape[0]
        d_mat = np.zeros((n_data, n_data))
        for i in range(n_data):
            d_mat[i, :] = (X != X[i]).sum(axis=1)
    else:
        print("距離行列を求めるさいにそのmetricsは指定できません。")
        exit(1)
    
    n_clusters = k
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(d_mat)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    print("クラスタラベル:", labels)
    print("クラスタの重心:\n", centroids)
    # pdb.set_trace()
    return labels

def custom_d(p1, p2, vector_alpha=0.8):
    spatial_d = np.linalg.norm(p1[:3] - p2[:3])
    vector_d = np.linalg.norm(p1[3:] - p2[3:])
    return vector_alpha * vector_d + (1 - vector_alpha) * spatial_d

def dbscan_vec_spac(label_history, pc):
    """ DBCSANの距離関数として、ベクトル間の距離と空間的な距離の両方を用いるように変更したもの。
    Args:
        label_history(ndarray): (点の数, ラベル遷移ベクトル)の行列。
        pc(ndarray): (点の数, 3)の行列。とりあえず0番目のフレームの点群を与える。
    Return:
        ndarray: 各点のラベル
    """
    label_history = label_history.T #(6890, 18)
    data = np.concatenate((pc, label_history), axis=1)
    eps = 0.1
    min_samples = 50

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=lambda p1, p2: custom_d(p1, p2))
    dbscan.fit(data)

    labels = dbscan.labels_
    print("クラスタラベル:", labels)
    return labels

def dbscan(X):
    X = X.T
    eps = 0.1
    min_samples = 30

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)

    labels = dbscan.labels_
    print("クラスタラベル:", labels)
    return labels
    

def hierarchical_clustering(X):
    X = X.T
    # 階層的クラスタリングのためのlinkage matrixの計算
    Z = linkage(X, method='ward')  # 'ward', 'complete', 'average', 'single'
    
    # デンドログラム
    plt.figure(figsize=(10, 7))
    plt.title("Dendrogram")
    dendrogram(Z)
    plt.savefig("tmp.png")

    max_d = 80.0  # クラスタを切り取る距離
    labels = fcluster(Z, max_d, criterion='distance')

    print("クラスタラベル:", labels)
    return labels


if __name__ == "__main__":
    # labels_path = "result/A04_pc_array/frames/0_188_10_0.001_withDBSCAN_eps0.1N5_NoizeSameLabel/label_history.pkl"
    labels_path = "result/B01_pc_array/frames/0_295_10_0.001_withDBSCAN/label_history.pkl"
    # pc_array_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A04/A04_pc_array.pkl"
    pc_array_path = "data/pointclouds/bodyHands_REGISTRATIONS_B01/B01_pc_array.pkl"
    # output_dir = "./result/A04_pc_array/frames/0_188_10_0.001_withDBSCAN_eps0.1N5_NoizeSameLabel/"
    output_dir = "result/B01_pc_array/frames/0_295_10_0.001_withDBSCAN/"

    file_name = "segmentated_"
    darray2video = NDARRAY2VIDEO(pc_array_path, output_dir, dir_rm=False)
    
    with open(labels_path, "rb") as f:
        label_history = pickle.load(f)
    with open(pc_array_path, "rb") as f:
        pc = pickle.load(f)
    
    
    preprocessing = ["whitening", "normalization"]
    # preprocessing = "normalization"
    preprocessing = "whitening"
    preprocessing = ["whitening"]
    preprocessing = None

    # clustering = "k-means"
    clustering = "my_k-medoids"
    # clustering = "k-means_custom"
    # clustering = "my_k-means"
    # clustering = "dbscan"
    # clustering = "dbscan_vec_spac"

    if preprocessing == None:
        file_name += "None"
    else:
        for pre in preprocessing:
            if pre == "whitening":
                label_history = whitening(label_history)
                file_name += "White"
            if pre == "normalization":
                label_history = normalize(label_history)
                file_name += "Norm"
    
    if clustering == "k-means":
        k =12
        labels = k_means(label_history, k)
        file_name += f"_k-means{k}"
        # mse = []
        # for k in range(5, 25):
        #     kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        #     kmeans.fit(label_history.T)
        #     labels = kmeans.labels_
        #     mse.append(kmeans.inertia_)
    elif clustering == "my_k-means":
        # metrics = "l0"
        # k = 12
        # mykmeans = MyKMeans(k, metrics=metrics, random_seed=0)
        # labels = mykmeans.fit(label_history.T)
        # file_name += f"_MyK-means{k}_{metrics}"
        metrics = "l0l2"
        k = 12
        mykmeans = MyKMeans(metrics=metrics, random_seed=7)
        labels = mykmeans.fit(k=k, X=label_history.T, pc=pc[0], labelVec_weight=0.7)
        file_name += f"_MyK-means{k}_{metrics}"
    elif clustering == "my_k-medoids":
        # metrics = "l0"
        # k = 12
        # mykmeans = MyKMeans(k, metrics=metrics, random_seed=0)
        # labels = mykmeans.fit(label_history.T)
        # file_name += f"_MyK-means{k}_{metrics}"
        metrics = "l0"
        k = 18
        mykmedoids = MyKMedoids(metrics=metrics, random_seed=0)
        labels = mykmedoids.fit(k=k, X=label_history.T)
        file_name += f"_MyK-medoids{k}_{metrics}"
    elif clustering == "k-means_custom":
        metrics = "l0"
        k = 12
        labels = k_means_custom_d(label_history, k, metrics)
        file_name += f"_CustomK-means{k}_{metrics}"
    elif clustering == "dbscan":
        labels = dbscan(label_history)
        file_name += "_dbscan"
    elif clustering == "dbscan_vec_spac":
        labels = dbscan_vec_spac(label_history, pc[0])
        file_name += "_dbscanVecSpac"
    # labels = hierarchical_clustering(label_history)
    
    darray2video = NDARRAY2VIDEO(pc_array_path, output_dir, dir_rm=False)
    darray2video.create(labels=labels, frame_idx=[0], file_names=[file_name], create_video=False)

    # ラベルの保存
    with open(output_dir + file_name + ".pkl", "wb") as f:
        pickle.dump(labels, f)


    # label_history = label_history.T
    # pc = pc[0]

    # # パラメータ設定
    # n_clusters = 20  # 初期クラスタ数
    # eps = 0.2       # DBSCANのeps（空間的な距離制約）
    # min_samples = 5 # DBSCANの最小サンプル数

    # # 1. ベクトルに基づく初期クラスタリング (KMeans)
    # kmeans = KMeans(n_clusters=n_clusters)
    # initial_labels = kmeans.fit_predict(label_history)
    # darray2video.create(labels=initial_labels, frame_idx=[0], file_names=[f"segmentated_initial_labels"], create_video=False)
    # # pdb.set_trace()

    # # 2. DBSCANで空間的な制約を加えたクラスタリング
    # final_labels = np.full(initial_labels.shape, -1)  # 最終的なラベル (-1は未分類)

    # for cluster in range(n_clusters):
    #     # 同じ初期クラスタに属する点を抽出
    #     cluster_points = np.where(initial_labels == cluster)[0]
        
    #     # 該当クラスタの座標を抽出
    #     cluster_coords = pc[cluster_points]
        
    #     # DBSCANを実行して空間的に再クラスタリング
    #     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    #     spatial_labels = dbscan.fit_predict(cluster_coords)
        
    #     # 結果を最終的なラベルに反映
    #     for i, point_idx in enumerate(cluster_points):
    #         if spatial_labels[i] != -1:  # ノイズでなければ
    #             final_labels[point_idx] = cluster * 100 + spatial_labels[i]

    # # 結果のラベルを表示
    # print("最終的なラベル:", final_labels)
    # label_cnt = {}
    # for l in np.unique(final_labels):
    #     label_cnt[int(l)] = int(sum(final_labels==l))
    # print(label_cnt)
    # # pdb.set_trace()
    # final_labels = np.arange(final_labels.shape[0])
    # darray2video.create(labels=final_labels, frame_idx=[0], file_names=[f"segmentated"], create_video=False)