##################################################################################################
## ラベル遷移ベクトルをクラスタリングすることで、物体表面点群のセグメンテーションを行う。
## 
##
##################################################################################################

import pickle, pdb
import numpy as np
from sklearn.cluster import KMeans
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
    """各点のラベル遷移ベクトルのL2ノルムが1になるように正規化"""
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    X_normalized = X / norms
    return X_normalized

def k_means(X, k):
    """sklearnのk-means++クラスタリング。距離尺度はL2に限定される
    Args:
        X(ndarray): (n_features, n_points)の形状
        k(int): クラスタ数
    Returns:
        labels(ndarray): ラベル
    """
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


if __name__ == "__main__":
    """ラベル遷移ベクトルのクラスタリングを行う。
    論文内の、「剛体変換に基づく手法」を行う場合は、ラベル遷移ベクトルの前処理なしで、k-medoidsで行えば良い。
    """
    
    ## 以下の４つのデータから適当に選択。異なるデータについて行う場合は適宜変更。
    labels_path = "result/A04_pc_array/frames/0_188_10_0.001_withDBSCAN_eps0.1N5_NoizeSameLabel/label_history.pkl"
    pc_array_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A04/A04_pc_array.pkl"
    output_dir = "./result/A04_pc_array/frames/0_188_10_0.001_withDBSCAN_eps0.1N5_NoizeSameLabel/"

    # labels_path = "./result/A11_pc_array/frames/0_793_10_0.001_withDBSCAN/label_history.pkl"
    # pc_array_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A11/A11_pc_array.pkl"
    # output_dir = "./result/A11_pc_array/frames/0_793_10_0.001_withDBSCAN/"

    # labels_path = "./result/SAPIEN_101387_pc_array/frames/0_60_5_0.001_withDBSCAN/label_history.pkl"
    # pc_array_path = "./data/SAPIEN/101387/SAPIEN_101387_pc_array.pkl"
    # output_dir = "./result/SAPIEN_101387_pc_array/frames/0_60_5_0.001_withDBSCAN/"
    
    # labels_path = "./result/SAPIEN_8961_pc_array/frames/0_60_5_0.001_withDBSCAN/label_history.pkl"
    # pc_array_path = "./data/SAPIEN/8961/SAPIEN_8961_pc_array.pkl"
    # output_dir = "./result/SAPIEN_8961_pc_array/frames/0_60_5_0.001_withDBSCAN/"


    file_name = "segmentated_"
    darray2video = NDARRAY2VIDEO(pc_array_path, output_dir, dir_rm=False)
    
    with open(labels_path, "rb") as f:
        label_history = pickle.load(f)
    with open(pc_array_path, "rb") as f:
        pc = pickle.load(f)
    
    ## ラベル遷移ベクトルに対する前処理。リストの先頭から順番に行う。前処理が不要な場合はNone。
    # preprocessing = ["whitening"]
    preprocessing = None

    ## ラベル遷移ベクトルのクラスタリング方法選択。
    ## Frame by Frameでのラベル付けの段階で空間的制約(DBSCAN)を行っているのであれば、"my_k-medoids"。
    ## "my_k-means"は、空間的制約を事前に与えず、クラスタリングの段階で三次元座標とラベル遷移ベクトルのクラスタリング結果を重みによって足し合わせて行う方法。
    ## k-medoidsと同じような結果が得られるが、L0ノルムとL2ノルムを合わせるのは不適切と言う理由で非推奨。
    clustering = "my_k-medoids"


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
    elif clustering == "my_k-means":
        metrics = "l0l2"
        k = 12
        seed = 0
        mykmeans = MyKMeans(metrics=metrics, random_seed=seed)
        labels = mykmeans.fit(k=k, X=label_history.T, pc=pc[0], labelVec_weight=0.7)
        file_name += f"_MyK-means{k}_{metrics}_seed{seed}"
    elif clustering == "my_k-medoids":
        # metrics = "l0"
        # k = 12
        # mykmeans = MyKMeans(k, metrics=metrics, random_seed=0)
        # labels = mykmeans.fit(label_history.T)
        # file_name += f"_MyK-means{k}_{metrics}"
        metrics = "l0"
        k = 2
        seed = 0
        mykmedoids = MyKMedoids(metrics=metrics, random_seed=seed, plot_progress=False, pc_array_path=pc_array_path, output_dir=output_dir)
        labels = mykmedoids.fit(k=k, X=label_history.T)
        file_name += f"_MyK-medoids{k}_{metrics}_seed{seed}"
    
    ## セグメンテーション結果の出力。
    darray2video = NDARRAY2VIDEO(pc_array_path, output_dir, dir_rm=False)
    darray2video.create(labels=labels, frame_idx=[0], file_names=[file_name], create_video=False)

    ## ラベルの保存
    with open(output_dir + file_name + ".pkl", "wb") as f:
        pickle.dump(labels, f)