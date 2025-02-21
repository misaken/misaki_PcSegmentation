

# from tools.tool_plot import *

# pc_array_path = "./data/SAPIEN/101387/SAPIEN_101387_pc_array.pkl"
# # pc_array_path = "./data/SAPIEN/101387/point_clouds2.pkl"
# pc_array_path = "./data/SAPIEN/8961/SAPIEN_8961_pc_array.pkl"
# output_dir_path = "./result/SAPIEN/check"
# # pc_array_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A04/A04_pc_array.pkl"
# # output_dir_path = "./result/A04_pc_array/frames/original"
# darray2video = NDARRAY2VIDEO(pc_array_path, output_dir_path)
# darray2video.create()


import open3d as o3d
import pdb, pickle
import numpy as np
from tools.tool_graph import *

# pc_array_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A04/A04_pc_array.pkl"
# output_path = "./result/A04_pc_array/DVariance_allPoints/0_188_10/"
# mesh_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A04/0000.ply"

# pc_array_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A11/A11_pc_array.pkl"
# output_path = "./result/A11_pc_array/DVariance_allPoints/0_792_10/"
# mesh_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A11/0000.ply"

pc_array_path = "./data/SAPIEN/8961/SAPIEN_8961_pc_array.pkl"
output_path = "./result/SAPIEN_8961_pc_array/DVariance_allPoints/0_60_5/"
mesh_path = "./data/SAPIEN/8961/mesh.ply"

# pc_array_path = "./data/SAPIEN/101387/SAPIEN_101387_pc_array.pkl"
# output_path = "./result/SAPIEN_101387_pc_array/DVariance_allPoints/0_60_5/"
# mesh_path = "./data/SAPIEN/101387/mesh.ply"


with open(output_path+"var_distances.pkl", "rb") as f:
    var_distances = pickle.load(f)
sqrt_var_distances = np.sqrt(var_distances)
with open(pc_array_path, "rb") as f:
    pc = pickle.load(f)
n_frames = pc.shape[0]

mesh = o3d.io.read_triangle_mesh(mesh_path)
ver = np.asarray(mesh.vertices)
tri = np.asarray(mesh.triangles)


n_points = ver.shape[0]

########################
# メッシュの全ての辺を用いる.エッジの重みは偏差
A = np.zeros((n_points, n_points))

for t in tri:
    for i in range(3):
        for j in range(i + 1, 3):
            idx1, idx2 = t[i], t[j]
            A[idx1, idx2] = A[idx2, idx1] = sqrt_var_distances[idx1, idx2]+1.0e-8
sample_size = n_points
points_idx = range(sample_size)


A = (1/A)
A[A==np.inf] = 0

tmp = P = pc[0][points_idx]
pdb.set_trace()
# P = pc[0][points_idx]
P = ver
P = P[:, [0, 2, 1]]
P[:, 1] *= -1

plot_graph(A, P, output_path)
# ########################

########################
# # メッシュの全ての辺を用いる.エッジの重みは3*n_frameのベクトル間の距離
# first_frame = 0
# end_frame = n_frames -1 # 最終フレームのインデックス
# skip = 10

# A = np.zeros((n_points, n_points))

# for t in tri:
#     for i in range(3):
#         for j in range(i + 1, 3):
#             idx1, idx2 = t[i], t[j]
#             A[idx1, idx2] = A[idx2, idx1] = np.sqrt(((pc[range(first_frame, end_frame+1, skip), idx1] - pc[range(first_frame, end_frame+1, skip), idx2])**2).sum())
# sample_size = n_points
# points_idx = range(sample_size)
########################


#######################
# # 任意のエッジ分離す
# graph = defaultdict(set)

# for t in tri:
#     for i in range(3):
#         for j in range(i + 1, 3):
#             graph[t[i]].add(t[j])
#             graph[t[j]].add(t[i])

# pdb.set_trace()

# def gen_A_skip_edge(n_points, graph, edge_distance):
#     selected_points_idx = set()
#     A = np.zeros((n_points, n_points))

#     visited = [False] * n_points # 探索済みか否か

#     queue = deque([0])  # 基準点となるノードのキュー
#     visited[0] = True

#     while queue: # 基準点を根として、幅優先探索
#         current_node = queue.popleft() # 根とするノードのインデックス

#         bfs_queue = deque([(current_node, 0)])  # 未探索ノードのうち、(各ノードのインデックス, 根ノードからの距離)のキュー

#         while bfs_queue:
#             node, distance = bfs_queue.popleft()

#             for neighbor in graph[node]:
#                 if not visited[neighbor]:
#                     if distance + 1 == edge_distance:
#                         A[current_node, neighbor] = A[neighbor, current_node] = sqrt_var_distances[current_node, neighbor]
#                         selected_points_idx.add(current_node)
#                         selected_points_idx.add(neighbor)
#                         visited[neighbor] = True
#                         queue.append(neighbor)  # 次の根ノードとして追加
#                     elif distance + 1 < edge_distance:
#                         bfs_queue.append((neighbor, distance + 1)) # edge_distanceに達するまで探索を続ける
#                         visited[neighbor] = True

#     return A, list(selected_points_idx)


# max_distance = 4 # 何エッジ離れた点をサンプリングするか
# A, points_idx = gen_A_skip_edge(n_points, graph, max_distance)

# pdb.set_trace()

# A = A[points_idx][:, points_idx]
# A = sqrt_var_distances[points_idx][:, points_idx]
# sample_size = len(points_idx)
####################


####################
# # メッシュの簡素化
# def simplify_mesh_by_triangles(triangles, vertices, target_number_of_faces):
#     """
#     Simplifies a mesh based on the provided triangle and vertex information.

#     Parameters:
#         triangles (numpy.ndarray): Array of triangle indices (shape: Nx3).
#         vertices (numpy.ndarray): Array of vertex positions (shape: Mx3).
#         target_number_of_faces (int): Desired number of triangles in the simplified mesh.

#     Returns:
#         numpy.ndarray: Simplified triangle indices (shape: Kx3).
#         numpy.ndarray: Simplified vertex positions (shape: Lx3).
#     """
#     # Create a TriangleMesh object from the provided data
#     mesh = o3d.geometry.TriangleMesh()
#     mesh.vertices = o3d.utility.Vector3dVector(vertices)
#     mesh.triangles = o3d.utility.Vector3iVector(triangles)

#     # Perform mesh simplification
#     simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_faces)

#     # Extract the simplified triangle and vertex data
#     simplified_triangles = np.asarray(simplified_mesh.triangles)
#     simplified_vertices = np.asarray(simplified_mesh.vertices)

#     return simplified_triangles, simplified_vertices

# target_faces = int(tri.shape[0] / 3)
# tri, ver = simplify_mesh_by_triangles(tri, ver, target_faces)
# pdb.set_trace()

# sample_size = ver.shape[0]
# points_idx = range(sample_size)

# # メッシュの全ての辺を用いる
# A = np.zeros((sample_size, sample_size))

# for t in tri:
#     for i in range(3):
#         for j in range(i + 1, 3):
#             idx1, idx2 = t[i], t[j]
#             A[idx1, idx2] = A[idx2, idx1] = sqrt_var_distances[idx1, idx2]
############################

# A = (1 / var_distances)
# A = (1/A)
# A[A==np.inf] = 0 #分散が大きいものはnp.infにしているのでAにinfが残ることはないと思うが、足切り忘れ対策で残す。
# A[np.isnan(A)] = 0 # var_distancesのルートを取ったことで、小さすぎる値がnanになる。そのため、本来は1/Aでnanは無限になるはずなので、nanも同様に0に置換する。

# pdb.set_trace()
# # Aの要素の分布図
# tril_idx = np.tril_indices(sample_size, k=-1)
# plt.hist(A[tril_idx], bins=np.linspace(A[tril_idx].min(), A[tril_idx].max(), 30))
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.xlim(A[tril_idx].min(), A[tril_idx].max())
# plt.title('Histogram of Data')
# plt.savefig(output_path+"hist_A_tril.png")
# plt.close()

# # pdb.set_trace()

# # # 分散の足切りだけではクラスタリング結果がほとんどの点が同ラベルになってしまった。距離の平均でも足切り追加。
# # tril_avg_distances = avg_distances[tril_idx] # 分散隣接行列の下三角行列の対角成分を除いたもの
# # q = np.array([0.75])
# # q_v_avg = np.quantile(tril_var_distances, q)
# # A[avg_distances >= q_v_avg] = 0.0

# D = np.zeros([sample_size, sample_size])
# D[range(sample_size), range(sample_size)] = A.sum(axis=1)

# # SAPIENデータセットを用いた場合は、次数行列Dの対角成分に0が発生する。そのためラプラシアン行列の固有値分解ができないため、Dに微小な値を加える。
# D[range(sample_size), range(sample_size)] += 1.0e-9


# L = D-A

# # Dの対角成分の分布図見てみる
# plt.hist(D[(np.arange(n_points), np.arange(n_points))], bins=np.linspace(D[(np.arange(n_points), np.arange(n_points))].min(), D[(np.arange(n_points), np.arange(n_points))].max(), 30))
# plt.xlabel('Value')

# plt.ylabel('Frequency')
# plt.xlim(D[(np.arange(n_points), np.arange(n_points))].min(), D[(np.arange(n_points), np.arange(n_points))].max())
# plt.title('Histogram of Data')
# plt.savefig(output_path+"hist_D_tril.png")
# plt.close()


# pdb.set_trace()
# # 対称正規化ラプラシアンにする
# L = np.sqrt(np.linalg.inv(D)) @ L @ np.sqrt(np.linalg.inv(D))

# # Lの下三角行列の対角成分を除いた要素の分布図見てみる
# plt.hist(L[np.tril_indices(sample_size, k=-1)], bins=np.linspace(L[np.tril_indices(sample_size, k=-1)].min(), L[np.tril_indices(sample_size, k=-1)].max(), 30))
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.xlim(L[np.tril_indices(sample_size, k=-1)].min(), L[np.tril_indices(sample_size, k=-1)].max())
# plt.title('Histogram of Data')
# plt.savefig(output_path+"hist_normedL_tril.png")
# plt.close()

# with open(output_path+"L.pkl", "wb") as f:
#     pickle.dump(L, f)



# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
# L = torch.tensor(L).to(device)
# values, vectors = torch.linalg.eig(L)
# values = values.cpu().numpy()
# vectors = vectors.cpu().numpy()
# L = L.cpu().numpy()

# k = 3 #クラスタ数
# v_sorted_idx = np.argsort(values)
# x = vectors[v_sorted_idx[1]] #0を除く最小固有値に対応する固有ベクトル
# # x = vectors[v_sorted_idx[1: 1+k]] #0を除く最小固有値に対応する固有ベクトル
# x = x.real #Lは対称行列なので固有値、固有ベクトルは実数
# pdb.set_trace()


# # 固有ベクトルの値の分布を可視化
# points = ver[points_idx]
# points = points[:, [0, 2, 1]]
# points[:, 1] *= -1
# color_labels = (x-x.min()) / (x.max()-x.min())

# colors = plt.cm.hsv(color_labels)

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
#                 c=colors, s=0.5)
# norm = plt.Normalize(vmin=0, vmax=1)
# cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='hsv'), ax=ax)
# cbar.set_label('Hue (Label)')

# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.savefig(output_path+"eigen_vector_map.png", dpi=300)





# kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
# kmeans.fit(X=x.reshape([-1, 1]))
# # kmeans.fit(X=x.reshape([n_points, -1]))

# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
# print("クラスタラベル:", labels)
# print("クラスタの重心:\n", centroids)

# pdb.set_trace()
# # ラベルを色分けして描画するときに、すべての点のラベルが必要な作りになってる。
# # サンプリングされていない点はkラベルとして割り振る
# labels_tmp = np.zeros(n_points) + k
# labels_tmp[points_idx] = labels
# labels_tmp = labels_tmp.astype(int)

# with open(output_path+"labels.pkl", "wb") as f:
#     pickle.dump(labels, f)

# pdb.set_trace()

# darray2video = NDARRAY2VIDEO(pc_array_path, output_path, dir_rm=False)
# darray2video.create(labels=labels_tmp, frame_idx=[0], points_idx=list(points_idx), file_names=[f"spectral_clustering_{k}"], create_video=False)




# グラフを描画
# pdb.set_trace()
# P = pc[0][points_idx]
# P = P[:, [0, 2, 1]]
# P[:, 1] *= -1

# plot_graph(A, P, output_path)