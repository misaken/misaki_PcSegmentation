####################################
## その他いろいろ
####################################

import pickle, subprocess
import open3d as o3d
import numpy as np

def gen_iter_n(p: float=0.99, r_out: float=0.98) -> int:
    """
    サンプリングの繰り返し回数を求める
    
    Args:
        p: k回サンプリングして、少なくとも一回は外れ値なしのサンプルのみを得ることを保証する確率
        r_out: 全体に対して、外れ値が占める割合
    Returns:
        int: 繰り返し回数k
    """
    k = int(np.log(1-p)/np.log(1-(1-r_out)**3))
    return k

#点群データをndarrayにして、pickleで保存
def ply2ndarray(data_path, pkl_path):
    # dir_path = "./support_data/my_pointcloud/bodyHands_REGISTRATIONS_A01/"
    file_n = int(subprocess.run(f"ls -1 {data_path} | wc -l", shell=True, capture_output=True, text=True).stdout) - 1
    points_array = []

    for i in range(file_n):
        file_path = data_path + str(i).zfill(4) + ".ply"
        point_cloud = o3d.io.read_point_cloud(file_path)
        points = np.asarray(point_cloud.points)
        points_array.append(points)
    points_array = np.stack(points_array)
    with open(pkl_path, "wb") as f:
        pickle.dump(points_array, f)
    #print(points_array.shape)


def calc_error_gap(pc, labels):
    """ 最終的なラベルごとに剛体変換を得て、それから求めた次の時刻の座標と真の座標との差を計算。
    Args:
        pc(ndarray): 全時刻における点群データ。(n_frames, n_points, 3)
        labels: 各点のラベル。(n_points, 1)
    Return:
        float: error
    """
    n_labels = np.unique(labels)
    for l in range(n_labels):
        pc[labels == l]

        