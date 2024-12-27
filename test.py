#################################################################################################
## グラフの可視化

# import pickle, os, pdb, torch, time
# import numpy as np
# from tools.tool_clustering import *
# from sklearn.cluster import KMeans
# from tools.tool_plot import *
# from tools.tool_graph import *


# random_state = np.random.RandomState(0)

# pc_array_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A04/A04_pc_array.pkl"
# output_path = "./result/A04_pc_array/DVariance_allPoints/0_188_10/"

# with open(output_path+"avg_distances.pkl", "rb") as f:
#     avg_distances = pickle.load(f)
# with open(output_path+"var_distances.pkl", "rb") as f:
#     var_distances = pickle.load(f)
# with open(pc_array_path, "rb") as f:
#     pc = pickle.load(f)

# n_points = var_distances.shape[0]


# tril_idx = np.tril_indices(var_distances.shape[0], k=-1)
# tril_var_distances = var_distances[tril_idx] # 分散隣接行列の下三角行列の対角成分を除いたもの

# # 重みの隣接行列Aで分散の逆数を取るので、分散がしきい値以上の部分の結果が0になるようにするためにnp.infにする。
# # 分散が第2四分位以上の要素はnp.inf。平均の足切りは一旦放置。
# q = np.array([0.125])
# q_v = np.quantile(tril_var_distances, q)
# print(q_v)
# var_distances[var_distances >= q_v] = np.inf
# np.fill_diagonal(var_distances, np.inf) # 対角成分が0なので、1/var_distancesでエラーが出る。回避。

# A = (1 / var_distances)
# A[A==np.inf] = 0 #分散が大きいものはnp.infにしているのでAにinfが残ることはないと思うが、足切り忘れ対策で残す。

# random_idx = random_state.choice(range(n_points), size=50, replace=False)
# A = A[random_idx][:, random_idx]

# P = pc[0][random_idx]
# P = P[:, [0, 2, 1]]
# P[:, 1] *= -1

# plot_graph(A, P, output_path)

###################################################################################################


###################################################################################################
## ドロネー三角形分割を行い、凸法のメッシュを描画

# import numpy as np
# from scipy.spatial import Delaunay
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import pickle, pdb

# output_path = "./result/A04_pc_array/DVariance_allPoints/0_188_10/"
# pc_array_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A04/A04_pc_array.pkl"

# random_state = np.random.RandomState(1)


# with open(pc_array_path, "rb") as f:
#     pc = pickle.load(f)
# n_points = pc.shape[1]
# random_idx = random_state.choice(range(n_points), size=6890)
# points = pc[0][random_idx]
# points = points[:, [0, 2, 1]]
# points[:, 1] *= -1

# tri = Delaunay(points)

# surface_triangles = tri.convex_hull

# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', s=0.5, label='Points')

# for simplex in surface_triangles:
#     vertices = points[simplex]
#     tri_vertices = Poly3DCollection([vertices], alpha=0.3, edgecolor='k', color='lightblue')
#     ax.add_collection3d(tri_vertices)

# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# ax.legend()
# plt.title("3D Mesh Generated Using Delaunay Triangulation (Surface Only)")
# plt.savefig(output_path+"delaunay.png")
# plt.close()
###################################################################################################

# pyvistaでもドロネー分割できる
# import numpy as np
# import pyvista as pv

# # メッシュを構築（点群 → Delaunay三角分割）
# cloud = pv.PolyData(points)
# mesh = cloud.delaunay_3d()

# # 可視化
# plotter = pv.Plotter()
# plotter.add_mesh(mesh, show_edges=True, color="lightblue", opacity=0.7)
# plotter.add_points(points, color="red", point_size=5.0)
# plotter.show()
###################################################################################################

###################################################################################################
# 凹包のメッシュをαシェイプで
# import pickle, pdb
# import numpy as np
# from scipy.spatial import Delaunay
# from scipy.spatial.distance import euclidean
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import matplotlib.pyplot as plt

# def compute_alpha_shape(points, alpha):
#     """
#     Computes the α-shape of a set of points in 3D space.

#     Parameters:
#         points (np.ndarray): Input points as an (N, 3) array.
#         alpha (float): Alpha parameter to control the shape detail.

#     Returns:
#         list: A list of simplices (triangles or tetrahedra) that form the α-shape.
#     """
#     # Perform Delaunay triangulation
#     delaunay = Delaunay(points)
    
#     # Get tetrahedra and their circumradius
#     simplices = delaunay.simplices
#     alpha_simplices = []  # Filtered simplices for the α-shape

#     for simplex in simplices:
#         vertices = points[simplex]
#         # Compute the circumradius of the simplex
#         circumcenter, circumradius = compute_circumcircle(vertices)
#         if circumradius < alpha:
#             alpha_simplices.append(simplex)

#     return alpha_simplices

# def compute_circumcircle(vertices):
#     """
#     Computes the circumcenter and circumradius of a tetrahedron in 3D.

#     Parameters:
#         vertices (np.ndarray): A (4, 3) array of tetrahedron vertices.

#     Returns:
#         tuple: Circumcenter (np.ndarray) and circumradius (float).
#     """
#     # Compute the matrix for solving circumcenter
#     A = np.vstack([vertices.T, np.ones(4)])
#     b = np.sum(vertices ** 2, axis=1)
    
#     # Solve for circumcenter in homogeneous coordinates
#     # x = np.linalg.solve(A, np.append(b, 1))  # Make sure A is square
#     x = np.linalg.solve(A, b)  # Make sure A is square
#     circumcenter = x[:3] / 2  # Divide by 2 to get the center
#     circumradius = np.sqrt(np.sum((vertices[0] - circumcenter) ** 2))
#     return circumcenter, circumradius

# def plot_alpha_shape(points, alpha, alpha_simplices):
#     """
#     Plots the α-shape result using matplotlib in 3D.

#     Parameters:
#         points (np.ndarray): Input points as an (N, 3) array.
#         alpha_simplices (list): List of simplices forming the α-shape.
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot the points
#     ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b', marker='o', s=0.5)

#     # Plot the alpha shape simplices (triangles)
#     for simplex in alpha_simplices:
#         simplex_points = points[simplex]
#         # Create a polygon from the vertices of the simplex
#         poly = Poly3DCollection([simplex_points], color='r', alpha=0.3)
#         ax.add_collection3d(poly)

#     ax.set_xlim([-1, 1])
#     ax.set_ylim([-1, 1])
#     ax.set_zlim([-1, 1])
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.savefig("./result/A04_pc_array/DVariance_allPoints/0_188_10/"+"AlphaShape.png")

# # Example usage
# if __name__ == "__main__":
#     output_path = "./result/A04_pc_array/DVariance_allPoints/0_188_10/"
#     pc_array_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A04/A04_pc_array.pkl"
#     random_state = np.random.RandomState(1)
#     with open(pc_array_path, "rb") as f:
#         pc = pickle.load(f)
#     n_points = pc.shape[1]
#     random_idx = random_state.choice(range(n_points), size=300)
#     points = pc[0][random_idx]
#     points = points[:, [0, 2, 1]]
#     points[:, 1] *= -1

#     alpha = 2.0

#     # Compute the α-shape
#     alpha_shape = compute_alpha_shape(points, alpha)

#     # Print result
#     print(f"Number of simplices in the α-shape: {len(alpha_shape)}")
#     plot_alpha_shape(points, alpha, alpha_shape)


############################################################################################33
## pyplのalphashape

# import alphashape, pickle, pdb
# import open3d as o3d
# import numpy as np
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import matplotlib.pyplot as plt


# output_path = "./result/A04_pc_array/DVariance_allPoints/0_188_10/"
# pc_array_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A04/A04_pc_array.pkl"
# random_state = np.random.RandomState(1)
# with open(pc_array_path, "rb") as f:
#     pc = pickle.load(f)
# n_points = pc.shape[1]
# random_idx = random_state.choice(range(n_points), size=6890)
# points = pc[0][random_idx]
# points = points[:, [0, 2, 1]]
# points[:, 1] *= -1

# alpha = 22
# alpha_shape = alphashape.alphashape(points, alpha)
# print("aa")
# # トライメッシュ（面）と頂点を取得
# # Trimeshオブジェクトの頂点と面を取得
# vertices = np.array(alpha_shape.vertices)
# faces = np.array(alpha_shape.faces)
# pdb.set_trace()
# print("aaaa")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 点群をプロット
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='r', label='Points', s=0.5)

# # αシェイプの面をプロット
# mesh = Poly3DCollection(vertices[faces], alpha=0.25, facecolors='cyan', linewidths=0.2, edgecolors='r')
# ax.add_collection3d(mesh)

# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.savefig("./result/A04_pc_array/DVariance_allPoints/0_188_10/"+"AlphaShape.png", dpi=300)
###############################################################################################################################


########################################################################################################################
## meshを用いたグラフのスペクトラルクラスタリング

# import open3d as o3d
# import pdb, pickle, torch
# import numpy as np
# from sklearn.cluster import KMeans
# from tools.tool_plot import *
# from collections import defaultdict, deque

# pc_array_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A04/A04_pc_array.pkl"
# output_path = "./result/A04_pc_array/DVariance_allPoints/0_188_10/"
# with open(output_path+"var_distances.pkl", "rb") as f:
#     var_distances = pickle.load(f)
# sqrt_var_distances = np.sqrt(var_distances)


# mesh = o3d.io.read_triangle_mesh("./data/pointclouds/bodyHands_REGISTRATIONS_A04/0000.ply")
# ver = np.asarray(mesh.vertices)
# tri = np.asarray(mesh.triangles)

# n_points = ver.shape[0]

# ########################
# # # メッシュの全ての辺を用いる
# # A = np.zeros((n_points, n_points))

# # for t in tri:
# #     for i in range(3):
# #         for j in range(i + 1, 3):
# #             idx1, idx2 = t[i], t[j]
# #             A[idx1, idx2] = A[idx2, idx1] = sqrt_var_distances[idx1, idx2]
# ########################

# # 任意のエッジ分離れた点同士の隣接行列
# graph = defaultdict(set)

# for t in tri:
#     for i in range(3):
#         for j in range(i + 1, 3):
#             graph[t[i]].add(t[j])
#             graph[t[j]].add(t[i])

# pdb.set_trace()

# def gen_A_skip_edge(n_points, graph, edge_distance):
#     A = np.zeros((n_points, n_points))

#     visited = [False] * n_points # 探索済みか否か

#     queue = deque([0])  # 基準点となるノードのキュー
#     visited[0] = True

#     while queue: # 基準点を根として、幅優先探索
#         # pdb.set_trace()
#         current_node = queue.popleft() # 根とするノードのインデックス

#         bfs_queue = deque([(current_node, 0)])  # 未探索ノードのうち、(各ノードのインデックス, 根ノードからの距離)のキュー

#         while bfs_queue:
#             node, distance = bfs_queue.popleft()

#             for neighbor in graph[node]:
#                 if not visited[neighbor]:
#                     if distance + 1 == edge_distance:
#                         A[current_node, neighbor] = A[neighbor, current_node] = sqrt_var_distances[current_node, neighbor]
#                         visited[neighbor] = True
#                         queue.append(neighbor)  # 次の根ノードとして追加
#                     elif distance + 1 < edge_distance:
#                         bfs_queue.append((neighbor, distance + 1)) # edge_distanceに達するまで探索を続ける
#                         visited[neighbor] = True

#     return A

# max_distance = 3 # 何エッジ離れた点をサンプリングするか
# A = gen_A_skip_edge(n_points, graph, max_distance)

# pdb.set_trace()


# # A = (1 / var_distances)
# A = (1/A)
# A[A==np.inf] = 0 #分散が大きいものはnp.infにしているのでAにinfが残ることはないと思うが、足切り忘れ対策で残す。

# # すべての点ではなく、ランダムサンプルですくなくしてみる
# sample_size = n_points
# random_idx = np.random.choice(range(n_points), size=sample_size, replace=False)
# A = A[random_idx][:, random_idx]

# # # Aの要素の分布図
# # tril_idx = np.tril_indices(sample_size, k=-1)
# # plt.hist(A[tril_idx], bins=np.linspace(A[tril_idx].min(), A[tril_idx].max(), 30))
# # plt.xlabel('Value')
# # plt.ylabel('Frequency')
# # plt.xlim(A[tril_idx].min(), A[tril_idx].max())
# # plt.title('Histogram of Data')
# # plt.savefig(output_path+"hist_A_tril.png")
# # plt.close()

# # pdb.set_trace()

# # # 分散の足切りだけではクラスタリング結果がほとんどの点が同ラベルになってしまった。距離の平均でも足切り追加。
# # tril_avg_distances = avg_distances[tril_idx] # 分散隣接行列の下三角行列の対角成分を除いたもの
# # q = np.array([0.75])
# # q_v_avg = np.quantile(tril_var_distances, q)
# # A[avg_distances >= q_v_avg] = 0.0

# # D = np.zeros([n_points, n_points])
# # D[range(n_points), range(n_points)] = A.sum(axis=1)
# D = np.zeros([sample_size, sample_size])
# D[range(sample_size), range(sample_size)] = A.sum(axis=1)



# L = D-A

# # # Dの対角成分の分布図見てみる
# # plt.hist(D[(np.arange(n_points), np.arange(n_points))], bins=np.linspace(D[(np.arange(n_points), np.arange(n_points))].min(), D[(np.arange(n_points), np.arange(n_points))].max(), 30))
# # plt.xlabel('Value')
# # plt.ylabel('Frequency')
# # plt.xlim(D[(np.arange(n_points), np.arange(n_points))].min(), D[(np.arange(n_points), np.arange(n_points))].max())
# # plt.title('Histogram of Data')
# # plt.savefig(output_path+"hist_D_tril.png")
# # plt.close()

# # 対称正規化ラプラシアンにする
# # L = np.sqrt(np.linalg.inv(D)) @ L @ np.sqrt(np.linalg.inv(D))

# # Lの下三角行列の対角成分を除いた要素の分布図見てみる
# # plt.hist(L[np.tril_indices(sample_size, k=-1)], bins=np.linspace(L[np.tril_indices(sample_size, k=-1)].min(), L[np.tril_indices(sample_size, k=-1)].max(), 30))
# # plt.xlabel('Value')
# # plt.ylabel('Frequency')
# # plt.xlim(L[np.tril_indices(sample_size, k=-1)].min(), L[np.tril_indices(sample_size, k=-1)].max())
# # plt.title('Histogram of Data')
# # plt.savefig(output_path+"hist_normedL_tril.png")
# # plt.close()

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

# k = 12 #クラスタ数
# v_sorted_idx = np.argsort(values)
# x = vectors[v_sorted_idx[1]] #0を除く最小固有値に対応する固有ベクトル
# # x = vectors[v_sorted_idx[1: 1+k]] #0を除く最小固有値に対応する固有ベクトル
# x = x.real #Lは対称行列なので固有値、固有ベクトルは実数
# pdb.set_trace()


# points = ver[random_idx]
# points = points[:, [0, 2, 1]]
# points[:, 1] *= -1
# color_labels = (x-x.min()) / (x.max()-x.min())

# # MatplotlibのHSVカラーマップを使用してラベルを色に変換
# colors = plt.cm.hsv(color_labels)

# # 3D描画用の設定
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# # 点群を描画
# sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
#                 c=colors, s=0.5)

# # カラーバーを追加
# norm = plt.Normalize(vmin=0, vmax=1)  # ラベル範囲 (0~1)
# cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='hsv'), ax=ax)
# cbar.set_label('Hue (Label)')

# # ラベルと見た目の調整
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.savefig(output_path+"eigen_vector_map.png", dpi=300)

# kmeans = KMeans(n_clusters=12, init='k-means++', random_state=0)
# kmeans.fit(X=x.reshape([-1, 1]))
# # kmeans.fit(X=x.reshape([n_points, -1]))

# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
# print("クラスタラベル:", labels)
# print("クラスタの重心:\n", centroids)

# # ラベルを色分けして描画するときに、すべての点のラベルが必要な作りになってる。
# # サンプリングされていない点はkラベルとして割り振る
# labels_tmp = np.zeros(n_points) + k
# labels_tmp[random_idx] = labels
# labels_tmp = labels_tmp.astype(int)

# with open(output_path+"labels.pkl", "wb") as f:
#     pickle.dump(labels, f)

# pdb.set_trace()

# darray2video = NDARRAY2VIDEO(pc_array_path, output_path, dir_rm=False)
# darray2video.create(labels=labels_tmp, frame_idx=[0], points_idx=list(random_idx), file_names=["spectral_clustering_12"], create_video=False)




# def gen_A_skip_edge(n_points, graph, edge_distance):
#     selected_points_idx = set()
#     A = np.zeros((n_points, n_points))

#     visited = [False] * n_points  # 全体探索済みか否か

#     queue = deque([0])  # 基準点となるノードのキュー
#     visited[0] = True

#     while queue:  # 幅優先探索
#         current_node = queue.popleft()  # 根ノードのインデックス

#         # ローカル探索済みノードの追跡
#         local_visited = [-1] * n_points  # 初期値 -1（未探索）
#         local_visited[current_node] = 0

#         bfs_queue = deque([current_node])  # 幅優先探索キュー

#         while bfs_queue:
#             node = bfs_queue.popleft()

#             for neighbor in graph[node]:
#                 if local_visited[neighbor] == -1:  # ローカル未訪問
#                     local_visited[neighbor] = local_visited[node] + 1

#                     if local_visited[neighbor] == edge_distance:  # 指定距離に到達
#                         A[current_node, neighbor] = A[neighbor, current_node] = sqrt_var_distances[current_node, neighbor]
#                         selected_points_idx.add(current_node)
#                         selected_points_idx.add(neighbor)
#                         if not visited[neighbor]:  # グローバル未訪問ならキューに追加
#                             queue.append(neighbor)
#                             visited[neighbor] = True

#                     elif local_visited[neighbor] < edge_distance:  # 探索を継続
#                         bfs_queue.append(neighbor)

#     return A, list(selected_points_idx)
############################################################################################################################################################


from tools.tool_RigidTransform import *
import numpy as np
import pickle, pdb


pc_array_path = "./data/pointclouds/bodyHands_REGISTRATIONS_A04/A04_pc_array.pkl"
labels_path = "result/A04_pc_array/frames/0_188_10_0.001_withDBSCAN_eps0.1N5_NoizeSameLabel/segmentated_None_MyK-medoids12_l0.pkl"

with open(pc_array_path, "rb") as f:
    pc = pickle.load(f)
with open(labels_path, "rb") as f:
    labels = pickle.load(f)
labels_unique = np.unique(labels)

n_frames = pc.shape[0]
n_points = pc.shape[1]
print(f"n_frames={n_frames}")
# 各フレームの差分からクラスタリング
first_frame = 0
end_frame = n_frames # 最終フレームのインデックス
skip = 10
# t = 1.0e-3 #誤差範囲
batch_size = 10000
# n_clusters = 20 # 求める剛体変換の最大個数
sample_size = 3

label_history = [] # 各点のラベルの推移
n_all_iter = (end_frame - first_frame) // skip
iter_cnt = 1
mse_per_label = np.zeros(labels_unique.shape[0])
for i in range(first_frame, end_frame-skip, skip):
    print(f"*******************************    frame {i} and {i+skip}  ({iter_cnt}/{n_all_iter})     *****************************")
    sys.stdout.flush() # nohup.outに書き込むように、明示的にフラッシュ
    X = pc[i]
    # Y = pc[i+skip-1]
    Y = pc[i+skip]
    for l in labels_unique:
        l_idx = np.where(labels==l)[0]
        T = svd2T(X[l_idx], Y[l_idx])
        Y_pred = rigid_transform(T, X[l_idx])
        mse_per_label[l] += ((Y[l_idx] - Y_pred)**2).sum() / l_idx.shape[0]
    iter_cnt += 1
print(mse_per_label)

    # T = svd2T(X[])

# T = svd2T(X[zero_idx[most_fitted_inlier_idx]], Y[zero_idx[most_fitted_inlier_idx]])
# refined_Y_pred = rigid_transform(refined_T, X[zero_idx]) #再度求めた剛体変換を、ラベル0全てにかける