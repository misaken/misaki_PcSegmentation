import torch, pdb, time
import numpy as np
# 複数の行列を持つテンソルを作成（例: バッチサイズが5の3x3行列）
t = torch.arange(1000*4*4).reshape([1000, 4, 4]).to(torch.float).to(torch.device("cuda"))
x = torch.arange(1000*4*7000).reshape([1000, 4, 7000]).to(torch.float).to(torch.device("cuda"))
t_s = time.time()
re = torch.bmm(t, x)
t_e = time.time()
print(t_e - t_s)
print(re.shape)
T = np.arange(1000*4*4).reshape([1000,4, 4])
X = np.arange(1000*4*7000).reshape([1000, 4, 7000])
t_s = time.time()
RE = np.matmul(T, X)
t_e = time.time()
print(t_e - t_s)
print(RE.shape)

# 結果を表示
# print("U:\n", U)
# print("S:\n", S)
# print("Vh:\n", Vh)

# import numpy as np
# from numpy.linalg import svd, matrix_rank
# import open3d as o3d
# import subprocess
# import pdb, pickle, os, cv2, time, torch
# from matplotlib import pyplot as plt
# from sklearn.cluster import KMeans
# from scipy.spatial.distance import mahalanobis
# from scipy.spatial.distance import cdist
# from sklearn.preprocessing import StandardScaler

# # path = "./support_data/body_models/smplh/ACCAD/s001/EricCamper04_poses.npz"
# # a = np.load(path)
# # print(type(a))

# #点群データをndarrayにして、pickleで保存
# def ply2ndarray(data_path, pkl_path):
#     # dir_path = "./support_data/my_pointcloud/bodyHands_REGISTRATIONS_A01/"
#     file_n = int(subprocess.run(f"ls -1 {data_path} | wc -l", shell=True, capture_output=True, text=True).stdout) - 1
#     points_array = []

#     for i in range(file_n):
#         file_path = data_path + str(i).zfill(4) + ".ply"
#         point_cloud = o3d.io.read_point_cloud(file_path)
#         points = np.asarray(point_cloud.points)
#         points_array.append(points)
#     points_array = np.stack(points_array)
#     with open(pkl_path, "wb") as f:
#         pickle.dump(points_array, f)
#     #print(points_array.shape)

# #ndarrayから動画に
# class NDARRAY2VIDEO:
#     def __init__(self, point_array_path):
#         self.num_frames = None
#         self.num_points = None
#         self.point_array_path = point_array_path # 点群をndarrayに変換したもののpickleファイルのパス
#         self.color_labels_path = "color_labels.pkl"
#         self.output_dir = "frames"

#     def create_frame(self, points, frame_number):
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         if points.shape[1]==6:
#             ax.scatter(points[:, 0], -points[:, 2], points[:, 1], c=points[:, 3:], marker='o', s=1)
#         else:
#             ax.scatter(points[:, 0], -points[:, 2], points[:, 1], marker='o', s=1)
#         ax.set_xlim([-1, 1])
#         ax.set_ylim([-1, 1])
#         ax.set_zlim([-1, 1])
#         ax.set_title(f'Frame {frame_number}')
        
#         output_path = os.path.join(self.output_dir, f'frame_{frame_number:04d}.png')
#         plt.savefig(output_path)
#         plt.close()

#     def create_video_from_frames(self, video_path, frame_rate=30):
#         """Create a video from the saved frames."""
#         frame_files = sorted([f for f in os.listdir(self.output_dir) if f.startswith('frame_') and f.endswith('.png')])
        
#         first_frame_path = os.path.join(self.output_dir, frame_files[0])
#         frame = cv2.imread(first_frame_path)
#         height, width, layers = frame.shape

#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

#         for frame_file in frame_files:
#             frame_path = os.path.join(self.output_dir, frame_file)
#             frame = cv2.imread(frame_path)
#             video.write(frame)

#         video.release()
    
#     def add_colors(self, points, colors, labels):
#         points_with_colors = np.zeros((self.num_frames, self.num_points, 6)) #6は、xyz + rgb
#         points_with_colors[:, :, :3] = points
#         for i in range(self.num_points):
#             points_with_colors[:, i, 3:] = colors[labels[i], :]
#         print(points_with_colors.shape)
#         return points_with_colors
    
#     def hsv_to_rgb(self, h, s, v):
#         if s == 0.0:
#             return v, v, v
#         i = int(h * 6.0)  # assume int() truncates!
#         f = (h * 6.0) - i
#         p = v * (1.0 - s)
#         q = v * (1.0 - s * f)
#         t = v * (1.0 - s * (1.0 - f))
#         i = i % 6
#         if i == 0:
#             return v, t, p
#         if i == 1:
#             return q, v, p
#         if i == 2:
#             return p, v, t
#         if i == 3:
#             return p, q, v
#         if i == 4:
#             return t, p, v
#         if i == 5:
#             return v, p, q

#     # 任意の個数のグラデーションをもつ色を作る
#     def create_colors(self, num_colors):
#         # hsvが分割しやすいので、そっちでn分割したあとrgbにする
#         hues = np.linspace(0, 1, num_colors, endpoint=False)  # Hueを0から1まで20等分
#         self.colors = np.array([self.hsv_to_rgb(h, 1.0, 1.0) for h in hues])  # RGBに変換
#         print(self.colors)

#         #with open(self.color_labels_path, "wb") as f:
#         #    pickle.dump(colors, f)
    
#     # labels : 点群に対するラベルを与える。ndarrayの一次元配列。ラベルは連番。
#     def create(self, labels=None):
#         with open(self.point_array_path, "rb") as f:
#             points = pickle.load(f)
#         #points = points[:20, 2000:3000, :] # インデックスが隣接した点の順になっているかの確認のため
#         self.num_frames = points.shape[0]
#         self.num_points = points.shape[1]
#         if labels is not None:
#             self.create_colors(np.unique(labels).shape[0]) # self.colorsを作成
#             points = self.add_colors(points, self.colors, labels) # self.points_with_colorsを作成

#         subprocess.run(["rm", "frames", "-r"])
#         os.makedirs(self.output_dir)
#         for i in range(self.num_frames):
#             self.create_frame(points[i], i)
        
#         self.create_video_from_frames(video_path='output.mp4')





#         print("fin")


# def kmeans_clustering():
#     #------------------------
#     with open("points_array.pkl", "rb") as f:
#         points = pickle.load(f)
#     # movement_vectors = points[1:] - points[:-1]
#     movement_vectors = np.diff(points, axis=0)

#     movement_vectors = np.vstack([np.zeros((1, 6890, 3)), movement_vectors])

#     features = np.concatenate([points, movement_vectors], axis=-1)

#     # 特徴量を2次元に変換 (6890, 3360)
#     features_reshaped = features.transpose(1, 0, 2).reshape(6890, -1)
#     # 特徴量の標準化
#     scaler = StandardScaler()
#     features_standardized = scaler.fit_transform(features_reshaped)

#     n_clusters = 20
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0)
#     clusters = kmeans.fit_predict(features_standardized)

#     with open("cluster_labels.pkl", "wb") as f:
#         pickle.dump(clusters, f)


# def hsv_to_rgb(h, s, v):
#     if s == 0.0:
#         return v, v, v
#     i = int(h * 6.0)  # assume int() truncates!
#     f = (h * 6.0) - i
#     p = v * (1.0 - s)
#     q = v * (1.0 - s * f)
#     t = v * (1.0 - s * (1.0 - f))
#     i = i % 6
#     if i == 0:
#         return v, t, p
#     if i == 1:
#         return q, v, p
#     if i == 2:
#         return p, v, t
#     if i == 3:
#         return p, q, v
#     if i == 4:
#         return t, p, v
#     if i == 5:
#         return v, p, q

# # 任意の個数のグラデーションをもつ色を作る
# def create_colors(num_colors):
#     # hsvが分割しやすいので、そっちでn分割したあとrgbにする
#     hues = np.linspace(0, 1, num_colors, endpoint=False)  # Hueを0から1まで20等分
#     colors = np.array([hsv_to_rgb(h, 1.0, 1.0) for h in hues])  # RGBに変換
#     print(colors)
#     with open("color_labels.pkl", "wb") as f:
#         pickle.dump(colors, f)


# #---------------------------------------------------------------------
# #以下剛体変換求める系

# # 回転行列は、ロール・ピッチ・ヨーの順番(x, y, z)でかけている
# def gen_T(r,t):
#     rx, ry, rz = np.radians(r)
#     Rx = np.array([
#         [1, 0, 0],
#         [0, np.cos(rx), -np.sin(rx)],
#         [0, np.sin(rx), np.cos(rx)]
#     ])
    
#     Ry = np.array([
#         [np.cos(ry), 0, np.sin(ry)],
#         [0, 1, 0],
#         [-np.sin(ry), 0, np.cos(ry)]
#     ])
    
#     Rz = np.array([
#         [np.cos(rz), -np.sin(rz), 0],
#         [np.sin(rz), np.cos(rz), 0],
#         [0, 0, 1]
#     ])

#     R = Rz @ Ry @ Rx
#     T = np.eye(4)
#     T[:3, :3] = R
#     T[:3, 3] = np.array(t)
    
#     return T

# #pyplotの座標の表現が行ベクトルで各点を表しているので、列ベクトルに修正した後変換したものをもとの形で返す。
# def rigid_transform(T, X):
#     # Tは同次座標, Xはpyplotで描画する前提の点集合。
#     num_data = X.shape[0]
#     X_T = X.T
#     ones_vector = np.ones([1, num_data])
#     X_ex = np.concatenate([X_T, ones_vector], 0)
#     transformed = T @ X_ex
#     return transformed[:-1, ].T

# def svd2T(X, Y):
#     # pdb.set_trace()
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#     X = torch.tensor(X).to(device)
#     Y = torch.tensor(Y).to(device)
#     X = X.T
#     Y = Y.T
#     num_data = X.shape[1]
#     # X_centroid = (np.sum(X, axis=1) / num_data).reshape(3,1)
#     # Y_centroid = (np.sum(Y, axis=1) / num_data).reshape(3,1)
#     X_centroid = (torch.sum(X, dim=1) / num_data).reshape(3,1)
#     Y_centroid = (torch.sum(Y, dim=1) / num_data).reshape(3,1)
#     X_, Y_ = X - X_centroid, Y - Y_centroid
#     A = X_ @ Y_.T
#     A = A.to(device)
#     #u, s, v = svd(A)
#     u, s, v = torch.linalg.svd(A)
#     # pdb.set_trace()
    
#     # R = v @ u.T
#     R = v.T @ u.T
#     detR = torch.linalg.det(R)
#     #print(f"det(R) = {detR}  s={s} v={v}")
#     #回転行列の行列式が-1の場合は反射になっているので、その時の対処
#     #特異値の一つが0の場合は、その対応するvの符号を反転する
#     if -1.10000<=detR<=0.98888:
#         if sum(s <= 1.0e-10) != 1:
#             print("can not find R")
#         else:
#             reflect_index = torch.where(s <= 1.0e-10)
#             v[reflect_index] = v[reflect_index] * -1
#             R = v.T @ u.T
#     t = (Y_centroid - R @ X_centroid).reshape(3, )

#     T = torch.eye(4)
#     T[:3, :3] = R
#     T[:3, 3] = t

#     return T.numpy()

# def calc_f_norm(A, B):
#     return np.sqrt(((A-B)**2).sum())


# #def ransac(data)


# if __name__ == "__main__":
#     #create video from point clouds
#     # ndarray2video = NDARRAY2VIDEO()
#     # ndarray2video.create(create_frame_flg=True)


#     # # dir_path = "./support_data/my_pointcloud/bodyHands_REGISTRATIONS_A01/"
#     # # file_n = int(subprocess.run(f"ls -1 {dir_path} | wc -l", shell=True, capture_output=True, text=True).stdout) - 1
#     # # points_array = []

#     # # for i in range(file_n):
#     # #     file_path = dir_path + str(i).zfill(4) + ".ply"
#     # #     point_cloud = o3d.io.read_point_cloud(file_path)
#     # #     pdb.set_trace()
#     # #     points = np.asarray(point_cloud.points)
#     # #     points_array.append(points)
#     # # points_array = np.stack(points_array)
    
    
#     # #create_colors()
#     # with open("color_labels.pkl", "rb") as f:
#     #     colors = pickle.load(f)
#     # with open("cluster_labels.pkl", "rb") as f:
#     #     clusters = pickle.load(f)
#     # with open("points_array.pkl", "rb") as f:
#     #     points = pickle.load(f)
#     # pdb.set_trace()


#     #--------------------------------------------------------------------------------------
#     #ply2ndarray("./data/pointclouds/bodyHands_REGISTRATIONS_A01/", "./A01_pc_array.pkl")
#     with open("./A01_pc_array.pkl", "rb") as f:
#         pc = pickle.load(f)
#     n_frame = pc.shape[0]
#     n_points = pc.shape[1]
#     print(f"n_frame={n_frame}")

#     # 各フレームの差分からクラスタリング
#     n_frame = 100
#     skip = 25
#     for i in range(0, n_frame+1-skip, skip):
#     # for i in range(n_frame-1):
#         X = pc[i]
#         Y = pc[i+skip]
#         n_clusters = 5
#         # 各点が属するクラスターラベル.0がクラス無し
#         cluster_label = np.zeros([n_points,], dtype=int)
#         label=1
#         for j in range(n_clusters): #クラスター数だけ繰り返す。
#             print(f"start {j}th iteration. cluster_unique={np.unique(cluster_label)}")
#             zero_idx = np.where(cluster_label == 0)[0]
#             n_sample = 3
#             n_iter = 10000
#             batch_size = 1000
#             t = 1.0e-7 #誤差範囲
#             d = n_points//20 #誤差範囲t以下の点がd個以上であれば正しい剛体変換
#             # norm_list = np.empty(0)
#             t_s = time.time()
#             for k in range(n_iter):
#                 # クラス0の点からサンプリング
#                 random_idx = np.random.choice(zero_idx, n_sample, replace=False)
#                 T = svd2T(X[random_idx, :], Y[random_idx, :])
#                 #print(T)
#                 Y_pred = rigid_transform(T, X)
#                 norm = ((Y - Y_pred)**2).sum(axis=1)
#                 # np.append(norm_list, norm)
#                 inlier_idx = norm < t
#                 n_inlier = sum(inlier_idx)
#                 if n_inlier > d:
#                     #クラスが未定(0)のうち、inlierであったものを新しくラベル付けする
#                     cluster_label[inlier_idx * (cluster_label==0)] = label
#                     print(f"n_inlier={n_inlier}, d={d}")
#                     print(f"label={label},  iter={k}")
#                     label += 1
#                     # break
#             else:
#                 print("can not fix")
#             # print(f"avg = {norm_list.mean()}\ns^2 = {sum((norm_list - norm_list.mean())**2) / norm_list.shape[0]}")
#             t_e = time.time()
#             print(f"time : {t_e - t_s}")

#         print(np.unique(cluster_label))
    
#     darray2video = NDARRAY2VIDEO("A01_pc_array.pkl")
#     darray2video.create(labels=cluster_label)

    
#     # darray2video = NDARRAY2VIDEO("A01_pc_array.pkl")
#     # darray2video.create()



# #----------------------------
# import numpy as np
# import pdb

# zero_idx = np.arange(20)
# idx = np.array([np.random.choice(zero_idx, 3, replace=False) for _ in range(5)])
# print(idx)
# print(type(idx))


# import itertools
# import numpy as np
# a = np.arange(3)
# b = range(60)
# tmp = list(itertools.combinations(b, 3))[:10]
# print(tmp)