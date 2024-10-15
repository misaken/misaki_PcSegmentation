import os, pickle, pdb
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from tools import *
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

data_path = "result/frames_change_delta/0_589_10/label_history.pkl"
pc_array_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A01/A01_pc_array.pkl"
output_dir = "./result/frames_change_delta/0_589_10/"

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
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    X_normalized = X / norms
    return X_normalized

def k_means(X):
    X = X.T
    n_clusters = 20
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(X)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    print("クラスタラベル:", labels)
    print("クラスタの重心:\n", centroids)
    # pdb.set_trace()
    return labels

def dbscan(X):
    X = X.T
    eps = 0.2
    min_samples = 4

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
    with open(data_path, "rb") as f:
        label_history = pickle.load(f)
    label_history = whitening(label_history)
    # label_history = normalize(label_history)
    
    # labels = k_means(label_history)
    # labels = dbscan(label_history)
    labels = hierarchical_clustering(label_history)
    darray2video = NDARRAY2VIDEO(pc_array_path, output_dir, dir_rm=False)
    darray2video.create(labels=labels, frame_idx=[0], create_video=False)
    
    
    