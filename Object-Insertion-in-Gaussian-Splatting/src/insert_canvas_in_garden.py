import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from line_profiler import profile as lp
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from typing import List


@dataclass
class GsData:
    def __init__(self):

        self.sh_degrees: int
        self.xyz: np.ndarray  # [n, 3]
        self.opacities: np.ndarray  # [n, 1]
        self.features_dc: np.ndarray  # ndarray[n, 3, 1], or tensor[n, 1, 3]
        self.features_rest: np.ndarray  # ndarray[n, 3, 15], or tensor[n, 15, 3]; NOTE: this is features_rest actually!
        self.scales: np.ndarray  # [n, 3]
        self.rotations: np.ndarray  # [n, 4]

    
    def qvec2rotmat(self, qvec):
        return np.array([
            [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
            [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
            [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

    def rotmat2qvec(self, R):
        Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
        K = np.array([
            [Rxx - Ryy - Rzz, 0, 0, 0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
        eigvals, eigvecs = np.linalg.eigh(K)
        qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
        if qvec[0] < 0:
            qvec *= -1
        return qvec

    def quat_multiply(self, quaternion0, quaternion1):
        w0, x0, y0, z0 = np.split(quaternion0, 4, axis=-1)
        w1, x1, y1, z1 = np.split(quaternion1, 4, axis=-1)
        return np.concatenate((
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        ), axis=-1)

    def transform_shs(self, features, rotation_matrix):
        """
        https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2147223570
        """

        try:
            from e3nn import o3
            import einops
            from einops import einsum
        except:
            print("Please run `pip install e3nn einops` to enable SHs rotation")
            return features

        if features.shape[1] == 1:
            return features

        features = torch.from_numpy(features)
        rotation_matrix = torch.from_numpy(rotation_matrix).to(torch.float32)

        features = features.clone()

        shs_feat = features[:, 1:, :]

        ## rotate shs
        P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=shs_feat.dtype, device=shs_feat.device)  # switch axes: yzx -> xyz
        inversed_P = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=shs_feat.dtype, device=shs_feat.device)
        
        permuted_rotation_matrix = inversed_P @ rotation_matrix @ P
        rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix.cpu())

        # Construction coefficient
        D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
        D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
        D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)

        # rotation of the shs features
        one_degree_shs = shs_feat[:, 0:3]
        one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
        one_degree_shs = einsum(
            D_1,
            one_degree_shs,
            "... i j, ... j -> ... i",
        )
        one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
        shs_feat[:, 0:3] = one_degree_shs

        if shs_feat.shape[1] >= 4:
            two_degree_shs = shs_feat[:, 3:8]
            two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
            two_degree_shs = einsum(
                D_2,
                two_degree_shs,
                "... i j, ... j -> ... i",
            )
            two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
            shs_feat[:, 3:8] = two_degree_shs

            if shs_feat.shape[1] >= 9:
                three_degree_shs = shs_feat[:, 8:15]
                three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
                three_degree_shs = einsum(
                    D_3,
                    three_degree_shs,
                    "... i j, ... j -> ... i",
                )
                three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
                shs_feat[:, 8:15] = three_degree_shs

        
        return features.numpy()



    def load_array_from_plyelement(self, plyelement, name_prefix: str, required: bool = True):
            names = [p.name for p in plyelement.properties if p.name.startswith(name_prefix)]

            names = sorted(names, key=lambda x: int(x.split('_')[-1]))
            
            v_list = [np.asarray(plyelement[attr_name]) for idx, attr_name in enumerate(names)]

            return np.stack(v_list, axis=1)


    # @lp
    def load_from_ply(self, ply_file_path: str):
        plydata = PlyData.read(ply_file_path)
        vertex = plydata['vertex'].data
        self.sh_degrees = 3

        self.xyz = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
        self.opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        self.features_dc = np.vstack([plydata.elements[0]["f_dc_0"], 
                        plydata.elements[0]["f_dc_1"], 
                        plydata.elements[0]["f_dc_2"]]).T[..., np.newaxis]

        self.features_rest = self.load_array_from_plyelement(plydata.elements[0], 
                                                "f_rest_", required=False).reshape((self.xyz.shape[0], 
                                                                                    self.sh_degrees, -1))

        
        self.scales = self.load_array_from_plyelement(plydata.elements[0], "scale_") #scale
        self.rotations = self.load_array_from_plyelement(plydata.elements[0], "rot_") # quatenion
        

    
    def rescale(self, factor:float):
        self.xyz = self.xyz * factor
        # self.scales = self.scales * factor
        self.scales += np.log(factor)

    def save_to_ply(self, path: str, with_colors: bool = False):
        # os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self.xyz
        normals = np.zeros_like(xyz)
        f_dc = self.features_dc.reshape((self.features_dc.shape[0], -1))

        if self.sh_degrees > 0:
            f_rest = self.features_rest.reshape((self.features_rest.shape[0], -1))
        else:
            f_rest = np.zeros((f_dc.shape[0], 0))

        opacities = self.opacities
        scale = self.scales
        rotation = self.rotations

        def construct_list_of_attributes():
            attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            attributes += [f'f_dc_{i}' for i in range(f_dc.shape[1])]
            if self.sh_degrees > 0:
                attributes += [f'f_rest_{i}' for i in range(f_rest.shape[1])]
            attributes += ["opacity"]
            attributes += [f'scale_{i}' for i in range(scale.shape[1])]
            attributes += [f'rot_{i}' for i in range(rotation.shape[1])]
            return attributes

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
        attribute_list = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]

        if with_colors:
            from sh_utils import eval_sh
            rgbs = np.clip(eval_sh(0, self.features_dc, None) + 0.5, 0.0, 1.0)
            rgbs = (rgbs * 255).astype(np.uint8)

            dtype_full += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            attribute_list.append(rgbs)

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(attribute_list, axis=1)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    
    def rotate(self, rpy: List):
        rot = R.from_euler('xyz', rpy) 
        quaternion = rot.as_quat()
        rot_mat = self.qvec2rotmat(quaternion)
        
        # Apply rotation to points
        self.xyz = (rot_mat @ self.xyz.T).T
        
        # Update rotations using quaternion multiplication
        self.rotations = torch.nn.functional.normalize(
            torch.from_numpy(self.quat_multiply(self.rotations, quaternion))
        ).numpy()
        
        # Transform SH coefficients
        features = np.concatenate((self.features_dc, self.features_rest), 
                                  axis=2).transpose((0, 2, 1))
        features = self.transform_shs(features, rot_mat)
        self.features_rest = features[:, 1:, :].transpose((0, 2, 1))

    def translation(self, x: float, y: float, z: float):
        if x == 0. and y == 0. and z == 0.:
            return
        # Apply translation after rotation
        self.xyz = self.xyz + np.array([x, y, z])
    
    def deg2rad(self, rpy_deg):
        return [(np.pi/180)  * i  for i in rpy_deg]

if __name__ == "__main__":
    base_obj_fdr = "/home/somusan/dev-somusan/classical_cv/3d_vision/3dgs/dataset/trash/"
    base_scene_fdr = "/home/somusan/dev-somusan/classical_cv/3d_vision/3dgs/dataset/garder/"
    
    # Load and process the object PLY file
    obj_path = os.path.join(base_obj_fdr, "point_cloud.ply")
    obj_gs_data = GsData()
    obj_gs_data.load_from_ply(obj_path)

    # Process object (scale and rotate)
    scale_factor = 0.06
    obj_gs_data.rescale(scale_factor)
    
    rpy = [80, -180, 30]   # -193 for y worked y-185, inc z - > rotates the obj like falling face first
    rpy_rad = obj_gs_data.deg2rad(rpy)
    obj_gs_data.rotate(rpy_rad)

    x, y, z = 0.05, -0.3, 0.6 # z val -> high bring up obj # inc in y -> toward the center
    obj_gs_data.translation(x,y,z)

    # Load the scene PLY file
    scene_path = os.path.join(base_scene_fdr, "iteration_6999_clean.ply")  
    scene_gs_data = GsData()
    scene_gs_data.load_from_ply(scene_path)


    
    # Merge the data
    # Concatenate all attributes
    merged_gs = GsData()
    # xyz = (1000000, 3) + (123196, 3) = (1123196, 3)
    merged_gs.xyz = np.concatenate([scene_gs_data.xyz, obj_gs_data.xyz], axis=0) 
    # opacities = (1000000, 1) + (123196, 1) = (1123196, 1)
    merged_gs.opacities = np.concatenate([scene_gs_data.opacities, obj_gs_data.opacities], axis=0) 
    
    #  features_dc = (1000000, 3, 1) + (123196, 3, 1) = (1123196, 3, 1)
    merged_gs.features_dc = np.concatenate([scene_gs_data.features_dc, obj_gs_data.features_dc], axis=0)
    #  features_rest = (1000000, 3, 15) + (123196, 3, 15) = (1123196, 3, 15)
    merged_gs.features_rest = np.concatenate([scene_gs_data.features_rest, obj_gs_data.features_rest], axis=0)
    #  scales = (1000000, 3) + (123196, 3) = (1123196, 3)
    merged_gs.scales = np.concatenate([scene_gs_data.scales, obj_gs_data.scales], axis=0)
    #  rotations -> (1000000, 4) + (123196, 4) = (1123196, 4)
    merged_gs.rotations = np.concatenate([scene_gs_data.rotations, obj_gs_data.rotations], axis=0)
    merged_gs.sh_degrees = scene_gs_data.sh_degrees  # Assuming both have same SH degrees


    # Save the merged result
    output_path = os.path.join("process_3dgs_file", "garden_canvas_merged_v2.ply")
    merged_gs.save_to_ply(output_path, with_colors=True)
    
    print(f"Original scene points: {len(scene_gs_data.xyz)}")
    print(f"Object points: {len(obj_gs_data.xyz)}")
    print(f"Merged points: {len(merged_gs.xyz)}")
    print(f"Saved merged result to: {output_path}")

