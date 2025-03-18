###############################################################################################
## メッシュの構造は考えずに、完全無向グラフの隣接行列を用いrバージョン
##
## var_distances : 全フレームでの距離の分散を重みとする隣接行列
##事前に分散と平均距離がしきい値以上の2点間は重みを0とし、辺を切っておく。
##
## グラフスペクトラルクラスタリングの際に重みが小さい部分を切っていくので、
## var_distancesの各要素を逆数にしたものを隣接行列Aとする。
## このとき、分散が限りなく0に近い要素は逆数を取るとnp.infになるので、それらは0にしている。
################################################################################################

import pickle, os, pdb, torch, time
import numpy as np
from pprint import pprint
from tools.tool_clustering import *
from sklearn.cluster import KMeans
from tools.tool_plot import *

np.random.seed(0)

pc_array_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A04/A04_pc_array.pkl"
output_path = "./result/A04_pc_array/DVariance_allPoints/0_188_10/"

with open(output_path+"avg_distances.pkl", "rb") as f:
    avg_distances = pickle.load(f)
with open(output_path+"var_distances.pkl", "rb") as f:
    var_distances = pickle.load(f)

n_points = var_distances.shape[0]

tril_idx = np.tril_indices(var_distances.shape[0], k=-1)
tril_var_distances = var_distances[tril_idx] # 分散隣接行列の下三角行列の対角成分を除いたもの

# 分散の分布図を描画
plt.hist(tril_var_distances, bins=np.linspace(tril_var_distances.min(), tril_var_distances.max(), 30))
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.xlim(tril_var_distances.min(), tril_var_distances.max())
plt.title('Histogram of Data')
plt.savefig(output_path+"var_distances.png")
plt.close()

# pdb.set_trace()
# 重みの隣接行列Aで分散の逆数を取るので、分散がしきい値以上の部分の結果が0になるようにするためにnp.infにする。
# 分散が第2四分位以上の要素はnp.inf。平均の足切りは一旦放置。
q = np.array([0.125])
q_v_var = np.quantile(tril_var_distances, q) # 0.00046ぐらい。分散の分布見た感じ、値が小さすぎる。
q_v_var = 0.01
var_distances[var_distances >= q_v_var] = np.inf
np.fill_diagonal(var_distances, np.inf) # 対角成分が0なので、1/var_distancesでエラーが出る。回避。

A = (1 / var_distances)
A[A==np.inf] = 0 #分散が大きいものはnp.infにしているのでAにinfが残ることはないと思うが、足切り忘れ対策で残す。

# すべての点ではなく、ランダムサンプルですくなくしてみる
sample_size = 300
random_idx = np.random.choice(range(n_points), size=sample_size, replace=False)
pdb.set_trace()
A = A[random_idx][:, random_idx]

# pdb.set_trace()
# # Aの要素の分布図
# plt.hist(A[tril_idx], bins=np.linspace(A[tril_idx].min(), A[tril_idx].max(), 30))
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.xlim(A[tril_idx].min(), A[tril_idx].max())
# plt.title('Histogram of Data')
# plt.savefig(output_path+"hist_A_tril.png")
# plt.close()

pdb.set_trace()

# # 分散の足切りだけではクラスタリング結果がほとんどの点が同ラベルになってしまった。距離の平均でも足切り追加。
# tril_avg_distances = avg_distances[tril_idx] # 分散隣接行列の下三角行列の対角成分を除いたもの
# q = np.array([0.75])
# q_v_avg = np.quantile(tril_var_distances, q)
# A[avg_distances >= q_v_avg] = 0.0

# D = np.zeros([n_points, n_points])
# D[range(n_points), range(n_points)] = A.sum(axis=1)
D = np.zeros([sample_size, sample_size])
D[range(sample_size), range(sample_size)] = A.sum(axis=1)



L = D-A

# # Dの対角成分の分布図見てみる
# plt.hist(D[(np.arange(n_points), np.arange(n_points))], bins=np.linspace(D[(np.arange(n_points), np.arange(n_points))].min(), D[(np.arange(n_points), np.arange(n_points))].max(), 30))
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.xlim(D[(np.arange(n_points), np.arange(n_points))].min(), D[(np.arange(n_points), np.arange(n_points))].max())
# plt.title('Histogram of Data')
# plt.savefig(output_path+"hist_D_tril.png")
# plt.close()

# 対称正規化ラプラシアンにする
# L = np.sqrt(np.linalg.inv(D)) @ L @ np.sqrt(np.linalg.inv(D))

# Lの下三角行列の対角成分を除いた要素の分布図見てみる
# plt.hist(L[np.tril_indices(sample_size, k=-1)], bins=np.linspace(L[np.tril_indices(sample_size, k=-1)].min(), L[np.tril_indices(sample_size, k=-1)].max(), 30))
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.xlim(L[np.tril_indices(sample_size, k=-1)].min(), L[np.tril_indices(sample_size, k=-1)].max())
# plt.title('Histogram of Data')
# plt.savefig(output_path+"hist_normedL_tril.png")
# plt.close()

with open(output_path+"L.pkl", "wb") as f:
    pickle.dump(L, f)



if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
L = torch.tensor(L).to(device)
values, vectors = torch.linalg.eig(L)
values = values.cpu().numpy()
vectors = vectors.cpu().numpy()
L = L.cpu().numpy()

k = 12 #クラスタ数
v_sorted_idx = np.argsort(values)
x = vectors[v_sorted_idx[1]] #0を除く最小固有値に対応する固有ベクトル
# x = vectors[v_sorted_idx[1: 1+k]] #0を除く最小固有値に対応する固有ベクトル
x = x.real #Lは対称行列なので固有値、固有ベクトルは実数
pdb.set_trace()

kmeans = KMeans(n_clusters=12, init='k-means++', random_state=0)
kmeans.fit(X=x.reshape([-1, 1]))
# kmeans.fit(X=x.reshape([n_points, -1]))

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print("クラスタラベル:", labels)
print("クラスタの重心:\n", centroids)

# ラベルを色分けして描画するときに、すべての点のラベルが必要な作りになってる。
# サンプリングされていない点はkラベルとして割り振る
labels_tmp = np.zeros(n_points) + k
labels_tmp[random_idx] = labels
labels_tmp = labels_tmp.astype(int)

with open(output_path+"labels.pkl", "wb") as f:
    pickle.dump(labels, f)

pdb.set_trace()

darray2video = NDARRAY2VIDEO(pc_array_path, output_path, dir_rm=False)
darray2video.create(labels=labels_tmp, frame_idx=[0], points_idx=list(random_idx), file_names=["spectral_clustering_12"], create_video=False)