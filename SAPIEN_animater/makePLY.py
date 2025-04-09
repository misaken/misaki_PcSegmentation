####################################################################
## urdf形式のファイルを、点群とメッシュの情報を持つplyファイルに変換する。
## 8961のurdfファイルが、一部の属性が明記されておらず、urdfpyモジュールで扱えない。
## そのため、修正したmobility_for_makePLY.urdfを扱う。
####################################################################

import os
import trimesh
import numpy as np
import open3d as o3d
from urdfpy import URDF
import pdb

# urdf_path = "./101387/mobility.urdf"
# output_dir = "./101387"
urdf_path = "./8961/mobility_for_makePLY.urdf"
output_dir = "./8961"

robot = URDF.load(urdf_path)

all_vertices = []
all_triangles = []
vertex_offset = 0

for link in robot.links:
    for visual in link.visuals:
        if visual.geometry.mesh is not None:
            mesh_path = os.path.join(os.path.dirname(urdf_path), visual.geometry.mesh.filename)

            mesh = trimesh.load(mesh_path)

            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)

            all_vertices.append(vertices)
            all_triangles.append(faces + vertex_offset)

            vertex_offset += len(vertices)

all_vertices = np.vstack(all_vertices)
all_triangles = np.vstack(all_triangles)

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
mesh.triangles = o3d.utility.Vector3iVector(all_triangles)

ply_path = os.path.join(output_dir, "mesh.ply")
o3d.io.write_triangle_mesh(ply_path, mesh)