
import os, pickle, subprocess, cv2, pdb, torch, sys
import numpy as np
import open3d as o3d
from numpy.linalg import svd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import DBSCAN

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


#ndarrayから動画に
class NDARRAY2VIDEO:
    def __init__(self, point_array_path, output_dir_path, dir_rm=True):
        """ ndarrayをプロット、動画生成
        Args:
            point_array_path(str): 点群ndarrayのpklパス
            output_dir_path(str): 結果の出力パス
            dir_rm(bool): フォルダを作り直すか、上書きするか。Trueで作り直し。
        """
        self.num_frames = None
        self.num_points = None
        self.point_array_path = point_array_path # 点群をndarrayに変換したもののpickleファイルのパス
        # self.color_labels_path = "color_labels.pkl"
        self.output_dir = output_dir_path
        self.colors = None
        self.custom_lines = None
        
        if dir_rm:
            if os.path.exists(self.output_dir):
                y_n = input(f"{self.output_dir} is already exists. Overwrite? (y/n)    ")
                while y_n != "y" and y_n != "n":
                    y_n = input("enter y or n.    ")
                if y_n == "y":
                    subprocess.run(["rm", self.output_dir, "-r"])
                    os.makedirs(self.output_dir)
                elif y_n == "n":
                    sys.exit(0)
            else:
                os.makedirs(self.output_dir)

    def create_frame(self, points, frame_number, file_name=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if points.shape[1]==7:
            ax.scatter(points[:, 0], -points[:, 2], points[:, 1], c=points[:, 3:6], marker='o', s=1, label=points[:, 6])
        else:
            ax.scatter(points[:, 0], -points[:, 2], points[:, 1], marker='o', s=1)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_title(f'Frame {frame_number}')

        # 凡例表示
        fig.legend(handles=self.custom_lines, loc="upper left")
        if file_name != None:
            output_path = os.path.join(self.output_dir, f'{file_name}.png')
        else:
            output_path = os.path.join(self.output_dir, f'frame_{frame_number:04d}.png')
        plt.savefig(output_path)
        plt.close()

    def create_video_from_frames(self, video_path, frame_rate=30):
        """Create a video from the saved frames."""
        frame_files = sorted([f for f in os.listdir(self.output_dir) if f.startswith('frame_') and f.endswith('.png')])
        
        first_frame_path = os.path.join(self.output_dir, frame_files[0])
        frame = cv2.imread(first_frame_path)
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

        for frame_file in frame_files:
            frame_path = os.path.join(self.output_dir, frame_file)
            frame = cv2.imread(frame_path)
            video.write(frame)

        video.release()
    
    def add_colors(self, points, labels):
        # クラスタ0が残っている場合はlabelsをそのままself.colorsのインデックスとする。
        # クラスタ0がない場合は、labelsから1引いたものをインデックスにする。
        if 0 not in np.unique(labels):
            labels -= 1
        # points_with_colors = np.zeros((self.num_frames, self.num_points, 6)) #6は、xyz + rgb
        # points_with_colors[:, :, :3] = points
        # for i in range(self.num_points):
        #     points_with_colors[:, i, 3:] = self.colors[labels[i], :]
        # print(points_with_colors.shape)
        # return points_with_colors
        points_with_colors = np.zeros((self.num_frames, self.num_points, 7)) #6は、xyz + rgb
        points_with_colors[:, :, :3] = points
        for i in range(self.num_points):
            points_with_colors[:, i, 3:6] = self.colors[labels[i], :]
            points_with_colors[:, i, 6] = labels[i]
        print(points_with_colors.shape)
        return points_with_colors
    
    def hsv_to_rgb(self, h, s, v):
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

    # # 任意の個数のグラデーションをもつ色を作る
    # def create_colors(self, num_colors):
    #     # hsvが分割しやすいので、そっちでn分割したあとrgbにする
    #     hues = np.linspace(0, 1, num_colors, endpoint=False)  # Hueを0から1まで20等分
    #     self.colors = np.array([self.hsv_to_rgb(h, 1.0, 1.0) for h in hues])  # RGBに変換
    #     print(f"colors = \n{self.colors}")
    def create_colors(self, unique_labels):
        # hsvが分割しやすいので、そっちでn分割したあとrgbにする
        num_colors = unique_labels.shape[0]
        # self.custom_lines = [
        #     Line2D([0], [0], marker='o', color=[1, 0, 0], label='Label 0', markersize=10, linestyle='None'),
        #     Line2D([0], [0], marker='o', color=[0, 1, 0], label='Label 1', markersize=10, linestyle='None')
        # ]
        hues = np.linspace(0, 1, num_colors, endpoint=False)  # Hueを0から1まで20等分
        self.colors = np.array([self.hsv_to_rgb(h, 1.0, 1.0) for h in hues])  # RGBに変換
        if 0 in unique_labels: # matplotのlegendを設定
            self.custom_lines = [Line2D([0], [0], marker='o', color=self.colors[l], label=l, markersize=5, linestyle='None') for l in unique_labels]
        else:
            self.custom_lines = [Line2D([0], [0], marker='o', color=self.colors[l-1], label=l, markersize=5, linestyle='None') for l in unique_labels]
        print(f"colors = \n{self.colors}")

    # サンプリングした点を描画
    def plot_sampled_points(self, samples, label_idx):
        if self.colors is None:
            # self.create_colors(label_idx.shape[0]) # self.colorsを作成
            self.create_colors(label_idx)
        elif self.colors.shape[0] != label_idx.shape[0]: # すでに作ったself.colorsと、渡されたlabel_idxのサイズが異なる場合は再度生成
            self.create_colors(label_idx)
        # pdb.set_trace()
        # points = self.add_colors(samples, labels)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        # if points.shape[1]==6:
        #     ax.scatter(points[:, 0], -points[:, 2], points[:, 1], c=points[:, 3:], marker='o', s=1)
        # else:
        #     ax.scatter(points[:, 0], -points[:, 2], points[:, 1], marker='o', s=1)
        T_n = samples.shape[0] # 求めた剛体変換の個数
        sampled_p_n = samples.shape[1] # 剛体変換を求めるのに用いた点の数
        # pdb.set_trace()
        for i in range(T_n):
            if 0 in label_idx:
                ax.scatter(samples[i, :, 0], -samples[i, :, 2], samples[i, :, 1], c=[self.colors[i+1]], alpha=1, marker='o', s=10)
            else:
                ax.scatter(samples[i, :, 0], -samples[i, :, 2], samples[i, :, 1], c=[self.colors[i]], alpha=1,  marker='o', s=10)
        ax.set_title("sampled points")
        output_path = os.path.join(self.output_dir, "sampled_points.png")
        plt.savefig(output_path)
        plt.close()

    def create(self, labels: np.ndarray=None, frame_idx: list=None, points_idx: list=None, file_names: list=None, create_video: bool=True) -> None:
        """ 複数の点群から動画を生成
        Args:
            labels: 点に対する一次配列のラベル。Noneの場合はラベルなしで描画。
            frame_idx: 描画するフレームを指定。Noneの場合は全フレーム描画。
            points_idx: 描画する点のインデックスを指定。
            file_names: 描画するフレームごとのタイトル。frame_idxがNoneで無いときのみ利用可能。frame_idxと同じサイズ。
            create_video: 動画を生成するか否か。Falseの場合はフレームのみ生成。
        """
        with open(self.point_array_path, "rb") as f:
            points = pickle.load(f)
        self.num_frames = points.shape[0]
        self.num_points = points.shape[1]
        if labels is not None:
            if self.colors is None:
                self.create_colors(np.unique(labels)) # self.colorsを作成
            elif self.colors.shape[0] != np.unique(labels).shape[0]: # すでに作ったself.colorsと、渡されたlabel_idxのサイズが異なる場合は再度生成
                self.create_colors(np.unique(labels))
            points = self.add_colors(points, labels) # points_with_colorsを作成

        # subprocess.run(["rm", "frames", "-r"])
        # os.makedirs(self.output_dir)
        if frame_idx == None:
            for i in range(self.num_frames):
                self.create_frame(points[i], i)
        else:
            for i in range(len(frame_idx)):
                if file_names != None:
                    if points_idx != None:
                        self.create_frame(points[frame_idx[i]][points_idx], frame_idx[i], file_names[i])
                    else:
                        self.create_frame(points[frame_idx[i]], frame_idx[i], file_names[i])
                else:
                    if points_idx != None:
                        self.create_frame(points[frame_idx[i]][points_idx], frame_idx[i])
                    else:
                        self.create_frame(points[frame_idx[i]], frame_idx[i])
        
        if create_video:
            self.create_video_from_frames(video_path='output.mp4')
        print("fin")


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

# 任意の個数のグラデーションをもつ色を作る
def create_colors(num_colors):
    # hsvが分割しやすいので、そっちでn分割したあとrgbにする
    hues = np.linspace(0, 1, num_colors, endpoint=False)  # Hueを0から1まで20等分
    colors = np.array([hsv_to_rgb(h, 1.0, 1.0) for h in hues])  # RGBに変換
    print(colors)
    with open("color_labels.pkl", "wb") as f:
        pickle.dump(colors, f)


#---------------------------------------------------------------------
#以下剛体変換求める系

# 回転行列は、ロール・ピッチ・ヨーの順番(x, y, z)でかけている
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
    n_data = X.shape[0]
    X_T = X.T
    ones_vector = np.ones([1, n_data])
    X_ex = np.concatenate([X_T, ones_vector], 0)
    transformed = T @ X_ex
    return transformed[:-1, ].T

#pyplotの座標の表現が行ベクトルで各点を表しているので、列ベクトルに修正した後変換したものをもとの形で返す。
def rigid_transform_batch(T, X):
    batch_size = T.shape[0]
    # Tは同次座標, Xはpyplotで描画する前提の点集合。
    n_data = X.shape[0]
    X_T = X.T
    ones_vector = np.ones([1, n_data])
    X_ex = np.concatenate([X_T, ones_vector], 0) # 同次座標系にする
    X_ex = X_ex.reshape([1, 4, n_data]).repeat(batch_size, axis=0)
    transformed = (T @ X_ex)[:, :-1, :].transpose(0, 2, 1)
    return transformed

def svd2T(X, Y):
    # pdb.set_trace()
    X = X.T
    Y = Y.T
    sample_size = X.shape[1]
    X_centroid = (np.sum(X, axis=1) / sample_size).reshape(3,1)
    Y_centroid = (np.sum(Y, axis=1) / sample_size).reshape(3,1)
    X_, Y_ = X - X_centroid, Y - Y_centroid
    A = X_ @ Y_.T
    u, s, v = svd(A)
    
    # R = v @ u.T
    R = v.T @ u.T
    detR = np.linalg.det(R)
    #print(f"det(R) = {detR}  s={s} v={v}")
    #回転行列の行列式が-1の場合は反射になっているので、その時の対処
    #特異値の一つが0の場合は、その対応するvの符号を反転する
    if -1.10000<=detR<=0.98888:
        if sum(s <= 1.0e-10) != 1: # 正しい剛体変換が得られない場合はinlierが少なくなるだけなので放置
            print("can not find R")
        else:
            reflect_index = np.where(s <= 1.0e-10)
            v[reflect_index] = v[reflect_index] * -1
            R = v.T @ u.T
    t = (Y_centroid - R @ X_centroid).reshape(3, )

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def svd2T_batch(X:np.ndarray, Y: np.ndarray):
    """
    バッチを用いて特異値分解で剛体変換を求める
    Args:
        X,Y: (batch_size, 3, 3)
    Return:
        np.ndarray: (batch_size, 4, 4)
    """
    X = X.transpose(0, 2, 1)
    Y = Y.transpose(0, 2, 1)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)
    sample_size = X.shape[2]
    batch_size = X.shape[0]
    X_centroid = (torch.sum(X, dim=2) / sample_size).reshape(batch_size, 3, 1)
    Y_centroid = (torch.sum(Y, dim=2) / sample_size).reshape(batch_size, 3, 1)
    X_, Y_ = X - X_centroid, Y - Y_centroid
    A = torch.matmul(X_, Y_.transpose(2, 1))
    A = A.to(device)
    #u, s, v = svd(A)
    u, s, vT = torch.linalg.svd(A, full_matrices=True)
    # pdb.set_trace()
    
    # R = v.T @ u.T
    R = torch.bmm(vT.transpose(1, 2), u.transpose(1, 2))
    detR = torch.linalg.det(R)
    #print(f"det(R) = {detR}  s={s} vT={vT}")
    #回転行列の行列式が-1の場合は反射になっているので、その時の対処
    #特異値の一つが0の場合は、その対応するvの符号を反転する
    detR_int = torch.round(detR).to(torch.int)
    reflect_index = torch.where(detR_int == -1)[0]
    error_n = 0
    for i in reflect_index:
        if sum(s[i] <= 1.0e-10) != 1:
            # print("can not find R")
            error_n += 1
        else:
            sig_change_index = torch.where(s[i] <= 1.0e-10)
            vT[sig_change_index] = vT[sig_change_index] * -1
            # R = vT.T @ u.T
    R = torch.bmm(vT.transpose(1, 2), u.transpose(1, 2))
    t = (Y_centroid - R @ X_centroid).reshape(batch_size, 3)
    # t = (Y_centroid - R @ X_centroid)
    T = torch.stack([torch.eye(4) for _ in range(batch_size)])
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    print(f"could not find R : {error_n}") if error_n != 0 else None

    return T.numpy()


def calc_f_norm(A, B):
    return np.sqrt(((A-B)**2).sum())


# 基準点からの距離の近さの上位数割の範囲からサンプリング
def sampling_by_distance(pc, zero_idx, sample_size):
    p_center_idx = np.random.choice(zero_idx)
    d = np.linalg.norm(pc - pc[p_center_idx], axis=1)
    n_points = pc.shape[0]
    near_idx = np.argsort(d)[:int(n_points*0.1)]
    sample_idx = np.append(np.random.choice(near_idx, sample_size-1, replace=False), p_center_idx)
    return sample_idx

def sampling_by_distance_batch(pc, zero_idx, batch_size, sample_size):
    n_points = pc.shape[0]
    p_center_idx = np.random.choice(zero_idx, size=batch_size)
    d = np.linalg.norm(pc - pc[p_center_idx].reshape([batch_size, 1, 3]), axis=2) # d:(batch_size, n_data)
    near_idx = np.argsort(d)[:, :int(n_points*0.3)]
    sample_idx = np.append(np.array([np.random.choice(row, size=sample_size-1, replace=False) for row in near_idx]), p_center_idx.reshape([batch_size, 1]), axis=1)
    return sample_idx




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


def dbscan(X):
    eps = 0.1
    min_samples = 30

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)

    labels = dbscan.labels_
    print("クラスタラベル:", labels)
    return labels


class MyKMeans():
    def __init__(self, n_clusters, metrics, max_iter=100, random_seed=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_seed)
        self.metrics = metrics
    
    def custom_distance_old(self, X, y):
        if self.metrics == "l0":
            distance = (X != y).sum(axis=1, dtype=float)
        elif self.metrics == "l2":
            distance = ((X - y)**2).sum(axis=1, dtype=float)
        else:
            sys.exit("Could not fid such a metrics. Select another one.")
        return distance
    
    def custom_distance(self, labels, labels_xentroids, points, points_centroids):
        # distances = (labels != labels_xentroids.reshape([labels_xentroids.shape[0], 1, labels.shape[1]])).sum(axis=2, dtype=float)
        # ((points - points_centroids.reshape([points_centroids.shape[0], 1, points.shape[1]]))**2).sum(axis=2, dtype=float)
        pass

    def calc_distance(self, X, idx=None, Y=None, metrics=None):
        """行列Xと、X[idx]の各点との距離を求める
        Args:
            X(ndarray): (n_samples, n_features)の行列
            idx(list): Xのどのサンプルとの距離を計算するかを指定。X内のデータとの距離を求めるときに利用。この場合は、自身との距離はnp.infになる。idxかYのいずれか。
            Y(ndarray): (n_samples, n_features)の行列。Yのサンプルサイズごとに、Xとの距離を求める。idxかYのいずれか。
            pc(ndarray): metricsが"l0l2"の場合のみ。三次元点群の座標。(n_samples, 3)
            pc_c(ndarray): metricsが"l0l2"の場合のみ。三次元でのクラスタ中心。(n_clusters, 3)
        Return:
            ndarray: (idxもしくはYのサンプルサイズ, Xのサンプルサイズ)の行列。各行は、Xの全データと、X[ixd]もしくはYの各サンプルとの距離をもつ。
        """
        if metrics is None: # metricsが指定されていない場合は、initでのmetricsを指定
            metrics = self.metrics
        
        if idx is None and Y is None:
            print("距離の比較対象がありません。idxかYを与えてください。")
            sys.exit(1)
        elif idx is not None and Y is not None:
            print("idxとYの両方が渡されています。どちらか一方のみにしてください。")
            sys.exit(1)
        elif idx is not None:
            if metrics == "l0":
                distances = (X != X[idx].reshape([len(idx), 1, X.shape[1]])).sum(axis=2, dtype=float)
            elif metrics == "l2":
                distances = ((X - X[idx].reshape([len(idx), 1, X.shape[1]]))**2).sum(axis=2, dtype=float)
            else:
                print("Could not fid such a metrics. Select another one.")
                sys.exit(1)
            distances[np.arange(len(idx)), idx] = np.inf
        elif Y is not None:
            if metrics == "l0":
                distances = (X != Y.reshape([Y.shape[0], 1, X.shape[1]])).sum(axis=2, dtype=float)
            elif metrics == "l2":
                distances = ((X - Y.reshape([Y.shape[0], 1, X.shape[1]]))**2).sum(axis=2, dtype=float)
            else:
                print("Could not fid such a metrics. Select another one.")
                sys.exit(1)
        return distances

    def fit(self, X, pc=None):
        """calc labels
        Args:
            X(ndarray): (n_samples, n_features)
            pc(ndarray): (n_samples, 3)。metricsがl0l2のときのみ。三次元点群座標。
        Return:
            ndarray: labels per samples. (n_samples, )
        """
        if self.metrics == "l0l2" and pc is None:
            print("metricsが'l0l2'の場合は、引数pcに三次元空間座標を与えてください")
            sys.exit(1)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        alpha_labelVec = 0.2 # クラスタリングでラベル遷移ベクトルと三次元空間座標の両方を用いるときの重み
        alpha_spatial = 1.0 - alpha_labelVec

        if self.metrics == "l0" or self.metrics == "l2":
            # k-means++で初期値を決定
            centroids_idx = []
            centroids_idx.append(self.random_state.randint(n_samples)) # １つ目のクラスタ中心をランダムに選択
            for i in range(1, self.n_clusters):
                # distances = np.empty((0, n_samples), dtype=float)
                # for j in range(i): # 各クラスタ中心と、サンプル点との距離の行列を作成。(num_cluster, num_samples)の形状。
                #     distance_tmp = self.custom_distance_old(X, X[centroids_idx[j]])
                #     distance_tmp[centroids_idx[j]] = np.inf # クラスタ中心自身との距離が最小になるのを防ぐために、無限にしておく
                #     distances = np.vstack([distances, distance_tmp])
                distances = self.calc_distance(X=X, idx=centroids_idx)
                min_distances = distances.min(axis=0) # 各点の最近傍クラスタ中心までの距離
                mask = np.ones(n_samples, dtype=bool) # クラスタ中心以外のインデックスマスク
                mask[centroids_idx[:i]] = False
                pr = min_distances[mask] / min_distances[mask].sum() # 次に取る初期値の確率分布
                centroids_idx.append(int(self.random_state.choice(np.arange(n_samples)[mask], size=1, replace=False, p=pr)[0]))
        elif self.metrics == "l0l2":
            centroids_idx = []
            centroids_idx.append(self.random_state.randint(n_samples))
            for i in range(1, self.n_clusters):
                labelVec_distances = self.calc_distance(X=X, idx=centroids_idx, metrics="l0")
                min_label_distances = labelVec_distances.min(axis=0)
                spatial_distances = self.calc_distance(X=pc, idx=centroids_idx, metrics="l2")
                min_spatial_distances = spatial_distances.min(axis=0)

                min_distances = alpha_labelVec*min_label_distances + alpha_spatial*min_spatial_distances

                mask = np.ones(n_samples, dtype=bool) # クラスタ中心以外のインデックスマスク
                mask[centroids_idx[:i]] = False
                
                pr = min_distances[mask] / min_distances[mask].sum()
                centroids_idx.append(int(self.random_state.choice(np.arange(n_samples)[mask], size=1, replace=False, p=pr)[0]))
        else:
            print("そのようなmetricsは無効です")
            sys.exit(1)
        
        # クラスタ中心の初期値を描画 テストなので適当 点群のパスは引数Xに応じて修正して
        # ndarray2video = NDARRAY2VIDEO("./data/pointclouds/bodyHands_REGISTRATIONS_A01/A01_pc_array.pkl", "./", dir_rm=False)
        # ndarray2video.create(frame_idx=[0], points_idx=centroids_idx, file_names=["first_cluster_centroids"], create_video=False)
        
        # クラスタの更新
        past_labels = None
        labels = np.zeros(n_samples) - 1

        if self.metrics == "l0" or self.metrics == "l2":
            centroids = X[centroids_idx].astype(float)
        elif self.metrics == "l0l2":
            centroids_labelVec = X[centroids_idx].astype(float)
            centroids_spatial = pc[centroids_idx].astype(float)
        # centroids = X[centroids_idx]
        
        iter_cnt = 0
        while (past_labels != labels).sum() != 0 and iter_cnt < self.max_iter:
            past_labels = labels

            if self.metrics == "l0" or self.metrics == "l2":
                distances = self.calc_distance(X=X, Y=centroids)
                # ユークリッド距離であれば1番目で問題ないが、l2だと大量に同じ距離が出てくるので二番目。そんなに時間もかからないし、とりあえず2にしておく。(収束しない問題はあるが、、、)
                # 1. 距離が最小のクラスタ中心に各点を割り当てる。
                # labels = distances.argmin(axis=0)
                
                # 2. 同じ距離の点があるときはランダムに割り振るようにする。
                tmp_labels = []
                for c in range(distances.shape[1]):
                    tmp_labels.append(self.random_state.choice(np.where((distances == distances.min(axis=0))[:, c])[0]))
                labels = np.array(tmp_labels)

                # クラスタ中心を再計算
                for c in range(self.n_clusters):
                    cluster_mask = (labels == c)
                    centroid = X[cluster_mask].sum(axis=0) / cluster_mask.sum()
                    centroids[c] = centroid
                if self.metrics == "l0":
                    centroids = np.round(centroids) # 要素が異なる個数をカウントするので、小数だとまずい。全部ことなる要素になってしまう。偶数丸めで一桁に(型はfloat)。
            
            elif self.metrics == "l0l2":
                # 各クラスタ中心との距離を求める
                labelVec_distances = self.calc_distance(X=X, Y=centroids_labelVec, metrics="l0")
                spatial_distances = self.calc_distance(X=pc, Y=centroids_spatial, metrics="l2")
                distances = (alpha_labelVec * labelVec_distances) + (alpha_spatial * spatial_distances)
                # 1. 距離が最小のクラスタ中心に各点を割り当てる。
                # labels = distances.argmin(axis=0)
                
                # 2. 同じ距離の点があるときはランダムに割り振るようにする。
                tmp_labels = []
                for c in range(distances.shape[1]):
                    tmp_labels.append(self.random_state.choice(np.where((distances == distances.min(axis=0))[:, c])[0]))
                labels = np.array(tmp_labels)
                
                pdb.set_trace()
                # クラスタ中心を再計算
                for c in range(self.n_clusters):
                    cluster_mask = (labels == c)
                    # centroid = X[cluster_mask].sum(axis=0) / int(cluster_mask.sum())
                    # centroids[c] = centroid
                    centroid_labelVec = X[cluster_mask].sum(axis=0) / cluster_mask.sum()
                    centroids_labelVec[c] = centroid_labelVec
                    centroid_spatial = pc[cluster_mask].sum(axis=0) / cluster_mask.sum()
                    centroids_spatial[c] = centroid_spatial
                centroids_labelVec = np.round(centroids_labelVec) # ラベル遷移ベクトルの距離がl0なので、クラスタ中心も整数にする。
            
            
            print(iter_cnt, np.unique(labels))
            iter_cnt += 1
        return labels