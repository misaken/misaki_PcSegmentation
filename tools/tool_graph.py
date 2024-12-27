import pdb, pickle
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import networkx as nx

def plot_graph(A, P, output_path):
    """グラフを可視化
    Args:
        A(ndarray): グラフに対する隣接行列.(n_points, n_points)
        P(ndarray): グラフの各頂点の三次元座標. (n_points, 3)
    Return:
        None
    """
    cmap = cm.get_cmap('viridis')

    # NetworkXグラフを作成
    G = nx.Graph()
    n_points = P.shape[0]

    # ノードを追加
    for i in range(n_points):
        G.add_node(i, pos=P[i])

    # エッジを追加（隣接行列 A に基づく）
    for i in range(n_points):
        for j in range(i + 1, n_points):  # 上三角成分のみ処理
            if A[i, j] > 0:  # 隣接行列の値が正の場合にエッジを追加
                G.add_edge(i, j)

    # ノードの位置（3次元）
    pos = nx.get_node_attributes(G, 'pos')

    # 可視化
    # fig = plt.figure(figsize=(10, 7))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ノードを描画
    # xs, ys, zs = zip(*P)
    # ax.scatter(xs, ys, zs, c='black', s=0.05, label="Points")

    # エッジを描画
    for edge in G.edges:
        start, end = edge
        x = [P[start][0], P[end][0]]
        y = [P[start][1], P[end][1]]
        z = [P[start][2], P[end][2]]
        # ax.plot(x, y, z, c='blue', alpha=0.5)
        weight = A[start, end]  # エッジの重み
        # color = cmap(weight / A.max())  # 重みを正規化してカラーマップに適用
        color = plt.cm.gist_rainbow(weight / A.max())
        ax.plot(x, y, z, c=color, alpha=0.8, linewidth=0.3)  # 色で重みを表現
    norm = plt.Normalize(vmin=0, vmax=1)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='gist_rainbow'), ax=ax)
    cbar.set_label('matplotlib.cm.gist_rainbow')


    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    # ax.set_title("3D Point Cloud with Fully Connected Graph")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.legend()
    plt.savefig(output_path+"graph.png", dpi=500)
    plt.close()

def plot_hist(output_path, data):
    plt.hist(data, bins=np.linspace(data.min(), data.max(), 30))
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xlim(data.min(), data.max())
    plt.title('Histogram of Data')
    plt.savefig(output_path+"hist_A_tril.png")
    plt.close()