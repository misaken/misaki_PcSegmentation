#########################################
## 任意の個数のジョイントに対して、フレーム毎の回転角を指定
## 8961用
#########################################

import trimesh
import numpy as np
import os, pdb, pickle
import xml.etree.ElementTree as ET

def rotation_matrix_axis_angle(axis, angle):
    """
    指定された回転軸と回転角に基づいて回転行列を生成する。
    回転軸が反転している場合（例: (0, -1, 0)）、適切な方向に調整して計算。

    Parameters:
        axis (tuple or list or numpy.ndarray): 回転軸 (x, y, z) を表す3要素の配列
        angle (float): 回転角（ラジアン単位）

    Returns:
        numpy.ndarray: 3x3 の回転行列
    """
    # 回転軸を正規化
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    
    # 回転軸がゼロベクトルの場合エラー
    if norm == 0:
        raise ValueError("回転軸が不適切")
    
    axis = axis / norm 
    
    if axis[1] < 0:
        axis = -axis
        angle = -angle
    
    x, y, z = axis
    
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    
    R = np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*y*x + s*z, t*y*y + c,   t*y*z - s*x],
        [t*z*x - s*y, t*z*y + s*x, t*z*z + c]
    ])
    
    return R

def parse_urdf(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    joints_info = []

    link_mesh_files = {}

    for link in root.findall('link'):
        link_name = link.get('name')
        mesh_files = []

        for visual in link.findall('visual'):
            geometry = visual.find('geometry')
            if geometry is not None:
                mesh = geometry.find('mesh')
                if mesh is not None:
                    mesh_filename = mesh.get('filename')
                    if mesh_filename:
                        mesh_files.append(mesh_filename)

        if mesh_files:
            link_mesh_files[link_name] = mesh_files

    for joint in root.findall('joint'):
        joint_name = joint.get('name')
        joint_type = joint.get('type')
        
        origin = joint.find('origin')
        axis = joint.find('axis')

        origin_xyz = origin.get('xyz') if origin is not None else None
        origin_rpy = origin.get('rpy') if origin is not None else None
        axis_xyz = axis.get('xyz') if axis is not None else None

        parent_link = joint.find('parent').get('link')
        child_link = joint.find('child').get('link')

        joint_info = {
            'joint_name': joint_name,
            'joint_type': joint_type,
            'origin_xyz': origin_xyz,
            'origin_rpy': origin_rpy,
            'axis_xyz': axis_xyz,
            'parent_link': parent_link,
            'child_link': child_link
        }
        joints_info.append(joint_info)

    return joints_info, link_mesh_files

pdb.set_trace()

data_path = "../data/SAPIEN/8961/"

joints_info, link_mesh_files = parse_urdf(data_path + "mobility.urdf")

print("ジョイント")
for joint in joints_info:
    print(joint)

print("\nリンクと対応するメッシュファイル")
for link, mesh_files in link_mesh_files.items():
    print(f"リンク名: {link}, メッシュファイル: {mesh_files}")


rotate_joint = {1: 5.0, 2: 2.0} # ジョイントインデックス: angle per frame
joint_origin_axis = {1: {"origin": None, "axis": None}, 2: {"origin": None, "axis": None}}
joint_angle = {1: 0.0, 2: 0.0}

frames = 60
point_clouds = []
for f in range(frames):
    point_cloud = []
    # angle = np.pi * ((5.0 * f) / 180)
    joint_angle[1] = np.pi * ((rotate_joint[1] * f) / 180)
    joint_angle[2] = np.pi * ((rotate_joint[2] * f) / 180)
    for i in range(len(joints_info)):
        if i in rotate_joint.keys():
            joint_origin_axis[i]["origin"] = joints_info[i]["origin_xyz"]
            joint_origin_axis[i]["axis"] = joints_info[i]["axis_xyz"]
            joint_origin_axis[i]["origin"] = [float(i) for i in joint_origin_axis[i]["origin"].split()]
            joint_origin_axis[i]["axis"] = [float(i) for i in joint_origin_axis[i]["axis"].split()]
        child_link = joints_info[i]["child_link"]
        print(child_link)
        
        for mesh_file_path in link_mesh_files[child_link]:
            print(data_path + mesh_file_path)
            mesh = trimesh.load(data_path + mesh_file_path)
            vertices = mesh.vertices
            # vertices = mesh.sample(1000)
            if i in rotate_joint.keys():
                R = rotation_matrix_axis_angle(joint_origin_axis[i]["axis"], joint_angle[i])
                print(R)
                vertices = (vertices-joint_origin_axis[i]["origin"])@R + joint_origin_axis[i]["origin"]
            points = vertices
            point_cloud.extend(points)
    point_clouds.append(point_cloud)
point_clouds = np.array(point_clouds)


with open(data_path + "point_clouds.pkl", "wb") as f:
    pickle.dump(point_clouds, f)