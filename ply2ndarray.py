################################################################################################
# plyファイルの点群データをndarrayにしてpickleで保存
# < 実行方法 > コマンドラインから下記のように実行
# python ply2ndarray.py データパス plyファイル名
# python ply2ndarray.py "./data/pointclouds/bodyHands_REGISTRATIONS_A01/" "./A01_pc_array.pkl"
################################################################################################

import pickle, subprocess, sys
import numpy as np
import open3d as o3d

def ply2ndarray(data_path, pkl_path):
    file_n = int(subprocess.run(f"ls -1 {data_path} | wc -l", shell=True, capture_output=True, text=True).stdout) - 1
    points_array = []

    for i in range(file_n):
        file_path = data_path + str(i).zfill(4) + ".ply"
        point_cloud = o3d.io.read_point_cloud(file_path)
        points = np.asarray(point_cloud.points)
        points_array.append(points)
    points_array = np.stack(points_array)
    with open(data_path+pkl_path, "wb") as f:
        pickle.dump(points_array, f)

if __name__ == "__main__":
    n_args = len(sys.argv)
    if n_args != 3:
        print(f"enter 2 arvs. {n_args-1} args entered.")
        sys.exit(1)
    data_path = sys.argv[1]
    output_path = sys.argv[2]
    ply2ndarray(data_path, output_path)