import sapien.core as sapien
from sapien.utils.viewer import Viewer
import numpy as np
import pdb, pickle

# エンジンとレンダラーの初期化
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)

# シーンの作成
scene = engine.create_scene()
scene.set_timestep(1 / 100.0)

# URDFデータのロード
loader = scene.create_urdf_loader()
loader.fix_root_link = True
data_path = "./data/SAPIEN/179"
asset = loader.load_kinematic(data_path+"/mobility.urdf")

if not asset:
    raise RuntimeError(f"Failed to load URDF: {data_path+'/mobility.urdf'}")

# ライティングの設定
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)

# # カメラの設定
# near, far = 0.1, 10.0
# width, height = 640, 480
# camera = scene.add_camera(
#     name="camera",
#     width=width,
#     height=height,
#     fovy=np.deg2rad(35),
#     near=near,
#     far=far,
# )
# camera.set_pose(sapien.Pose(p=[1, 0, 0]))

# # カメラの位置を設定
# camera_mount_actor = scene.create_actor_builder().build_kinematic()
# camera.set_parent(parent=camera_mount_actor, keep_pose=False)
# cam_pos = np.array([-2, -2, 3])
# cam_pos = np.array([2, -2, 3])
# forward = -cam_pos / np.linalg.norm(cam_pos)
# left = np.cross([0, 0, 1], forward)
# left = left / np.linalg.norm(left)
# up = np.cross(forward, left)
# mat44 = np.eye(4)
# mat44[:3, :3] = np.stack([forward, left, up], axis=1)
# mat44[:3, 3] = cam_pos
# camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))

# カメラの設定
near, far = 0.1, 10.0
width, height = 480, 480
camera = scene.add_camera(
    name="camera",
    width=width,
    height=height,
    fovy=np.deg2rad(35),
    near=near,
    far=far,
)

# カメラ位置
cam_pos = np.array([-3, 0, 0])
target_pos = np.array([0, 0, 0])  # カメラが注視するポイント
forward = target_pos - cam_pos  # カメラの前方向
forward = forward / np.linalg.norm(forward)  # 正規化

# カメラの座標系を計算
up = np.array([0, 0, 1])  # カメラの「上方向」
right = np.cross(up, forward)  # カメラの「右方向」
right = right / np.linalg.norm(right)  # 正規化
up = np.cross(forward, right)  # 更新された「上方向」

# カメラの回転行列
rotation_matrix = np.stack([forward, right, up], axis=1)

# カメラの姿勢を設定
pose_matrix = np.eye(4)
pose_matrix[:3, :3] = rotation_matrix
pose_matrix[:3, 3] = cam_pos
camera.set_pose(sapien.Pose.from_transformation_matrix(pose_matrix))



# シーンの更新
scene.step()
scene.update_render()
camera.take_picture()

# カメラ位置からポイントクラウドを取得
position = camera.get_float_texture("Position")  # (H, W, 4)
points_opengl = position[..., :3]  # 最初の3チャネルを取得
# mask = position[..., 3] < 1  # 有効なポイントをフィルタリング
# points_opengl = points_opengl[mask]

# OpenGL空間からワールド空間に変換
model_matrix = camera.get_model_matrix()
points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]

# (n_points, 3) の形状に変換
points_world = points_world.reshape(-1, 3)

print(f"Point cloud shape: {points_world.shape}")  # (n_points, 3)

viewer = Viewer(renderer)
viewer.set_scene(scene)

while not viewer.closed:
    scene.step()
    scene.update_render()
    viewer.render()

points_world = points_world.reshape([1, -1, 3])
print(f"Point cloud shape: {points_world.shape}")  # (n_points, 3)

pdb.set_trace()

with open(data_path+"/pc.pkl", "wb") as f:
    pickle.dump(points_world, f)

# 結果: (n_points, 3) の点群データ