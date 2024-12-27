########################################
## 点群の描画関係
########################################

import sys, subprocess, pickle, pdb, os, cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


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
        plt.savefig(output_path, dpi=300)
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
        hues = np.linspace(0, 1, num_colors, endpoint=False)  # Hueを0から1までn等分
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
            points_idx: 描画する点のインデックスを指定。これを指定しても、labelsでは全点に対するラベルを与える必要がある。
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

# # 任意の個数のグラデーションをもつ色を作る
# def create_colors(num_colors):
#     # hsvが分割しやすいので、そっちでn分割したあとrgbにする
#     hues = np.linspace(0, 1, num_colors, endpoint=False)  # Hueを0から1まで20等分
#     colors = np.array([hsv_to_rgb(h, 1.0, 1.0) for h in hues])  # RGBに変換
#     print(colors)
#     with open("color_labels.pkl", "wb") as f:
#         pickle.dump(colors, f)