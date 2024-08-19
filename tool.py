import numpy as np
import open3d as o3d
import subprocess
import pdb, pickle, os, cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import mahalanobis
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# path = "./support_data/body_models/smplh/ACCAD/s001/EricCamper04_poses.npz"
# a = np.load(path)
# print(type(a))

#点群データをndarrayに
def ply2ndarray():
    dir_path = "./support_data/my_pointcloud/bodyHands_REGISTRATIONS_A01/"
    file_n = int(subprocess.run(f"ls -1 {dir_path} | wc -l", shell=True, capture_output=True, text=True).stdout) - 1
    points_array = []

    for i in range(file_n):
        file_path = dir_path + str(i).zfill(4) + ".ply"
        point_cloud = o3d.io.read_point_cloud(file_path)
        points = np.asarray(point_cloud.points)
        points_array.append(points)
    points_array = np.stack(points_array)
    with open("points_array.pkl", "wb") as f:
        pickle.dump(points_array, f)
    #print(points_array.shape)

#ndarrayから動画に
class NDARRAY2VIDEO:
    def __init__(self, n_frame=590, n_points=6890):
        self.num_frames = 590
        self.num_points = 6890

    def create_frame(self, points, frame_number, output_dir):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], -points[:, 2], points[:, 1], c=points[:, 3:], marker='o', s=1)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_title(f'Frame {frame_number}')
        
        output_path = os.path.join(output_dir, f'frame_{frame_number:04d}.png')
        plt.savefig(output_path)
        plt.close()

    def create_video_from_frames(self, output_dir, video_path, frame_rate=30):
        """Create a video from the saved frames."""
        frame_files = sorted([f for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('.png')])
        
        first_frame_path = os.path.join(output_dir, frame_files[0])
        frame = cv2.imread(first_frame_path)
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

        for frame_file in frame_files:
            frame_path = os.path.join(output_dir, frame_file)
            frame = cv2.imread(frame_path)
            video.write(frame)

        video.release()
    
    def add_colors(self, points, colors, clusters):
        points_with_colors = np.zeros((self.num_frames, self.num_points, 6))
        points_with_colors[:, :, :3] = points
        for i in range(self.num_points):
            points_with_colors[:, i, 3:] = colors[clusters[i], :]
        print(points_with_colors.shape)
        return points_with_colors

    def create(self, create_frame_flg=True):
        with open("points_array.pkl", "rb") as f:
            points = pickle.load(f)
        with open("color_labels.pkl", "rb") as f:
            colors = pickle.load(f)
        with open("cluster_labels.pkl", "rb") as f:
            clusters = pickle.load(f)
        
        points_with_colors = self.add_colors(points, colors, clusters)
        
        output_dir = 'frames'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if create_frame_flg:
            for frame_number in range(self.num_frames):
                self.create_frame(points_with_colors[frame_number], frame_number, output_dir)
        
        video_path = 'output.mp4'
        self.create_video_from_frames(output_dir, video_path)

        print("fin")


def kmeans_clustering():
    #------------------------
    with open("points_array.pkl", "rb") as f:
        points = pickle.load(f)
    # movement_vectors = points[1:] - points[:-1]
    movement_vectors = np.diff(points, axis=0)

    movement_vectors = np.vstack([np.zeros((1, 6890, 3)), movement_vectors])

    features = np.concatenate([points, movement_vectors], axis=-1)

    # 特徴量を2次元に変換 (6890, 3360)
    features_reshaped = features.transpose(1, 0, 2).reshape(6890, -1)
    # 特徴量の標準化
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features_reshaped)

    n_clusters = 20
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(features_standardized)

    with open("cluster_labels.pkl", "wb") as f:
        pickle.dump(clusters, f)


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)  # assume int() truncates!
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q

def create_colors():
    num_colors = 20
    hues = np.linspace(0, 1, num_colors, endpoint=False)  # Hueを0から1まで20等分
    colors = np.array([hsv_to_rgb(h, 1.0, 1.0) for h in hues])  # RGBに変換
    print(colors)
    with open("color_labels.pkl", "wb") as f:
        pickle.dump(colors, f)


#---------------------------------------------------------------------
#以下剛体変換求める系

# 回転行列は、ロール・ピッチ・ヨーの順番(x, y, z)
def gen_T(r,t):
    rx, ry, rz = np.radians(r)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array(t)
    
    return T

#pyplotの座標の表現が行ベクトルで各点を表しているので、列ベクトルに修正した後変換したものをもとの形で返す。
def rigid_transform(T, X):
    # Tは同次座標, Xはpyplotで描画する前提の点集合。
    num_data = X.shape[0]
    X_T = X.T
    ones_vector = np.ones([1, num_data])
    X_ex = np.concatenate([X_T, ones_vector], 0)
    transformed = T @ X_ex
    return transformed[:-1, ].T

def svd2T(X, Y):
    # pdb.set_trace()
    X = X.T
    Y = Y.T
    num_data = X.shape[1]
    X_centroid = (np.sum(X, axis=1) / num_data).reshape(3,1)
    Y_centroid = (np.sum(Y, axis=1) / num_data).reshape(3,1)
    X_, Y_ = X - X_centroid, Y - Y_centroid
    A = X_ @ Y_.T
    u, s, v = svd(A)
    # R = v @ u.T
    R = v.T @ u.T
    print(f"det(R) = {np.linalg.det(R)}")
    t = (Y_centroid - R @ X_centroid).reshape(3, )

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

def calc_f_norm(A, B):
    return np.sqrt(((A-B)**2).sum())


if __name__ == "__main__":
    #create video from point clouds
    ndarray2video = NDARRAY2VIDEO()
    ndarray2video.create(create_frame_flg=True)


    # dir_path = "./support_data/my_pointcloud/bodyHands_REGISTRATIONS_A01/"
    # file_n = int(subprocess.run(f"ls -1 {dir_path} | wc -l", shell=True, capture_output=True, text=True).stdout) - 1
    # points_array = []

    # for i in range(file_n):
    #     file_path = dir_path + str(i).zfill(4) + ".ply"
    #     point_cloud = o3d.io.read_point_cloud(file_path)
    #     pdb.set_trace()
    #     points = np.asarray(point_cloud.points)
    #     points_array.append(points)
    # points_array = np.stack(points_array)
    
    
    #create_colors()
    with open("color_labels.pkl", "rb") as f:
        colors = pickle.load(f)
    with open("cluster_labels.pkl", "rb") as f:
        clusters = pickle.load(f)
    with open("points_array.pkl", "rb") as f:
        points = pickle.load(f)
    pdb.set_trace()
