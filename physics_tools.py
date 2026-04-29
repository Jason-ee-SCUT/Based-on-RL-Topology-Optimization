# 工具库
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN
from scipy.interpolate import griddata
import trimesh
import networkx as nx
from skimage import measure

# 全局配置参数
GRID_SIZE = (100, 100, 150)
VOXEL_SIZE = 1e-4 # 0.1mm       
RHO_DENSITY = 19350.0  
CORES = 16

def initialize_cylinder_grid():
    nx, ny, nz = GRID_SIZE
    grid = np.zeros((nx, ny, nz), dtype=np.float32)
    radius = 25  
    x, y, z = np.ogrid[:nx, :ny, :nz]
    dist_sq = (x - nx//2)**2 + (y - ny//2)**2
    cylinder_mask = (dist_sq <= radius**2) & (z >= 0) & (z < nz)
    grid[cylinder_mask] = 1.0
    return grid

def calculate_evaporation_rate(temp_map):
    # 计算蒸发率图谱
    # 2000℃以上考虑蒸发
    A = 3.9e7       
    B = -1.023e5    
    evap_map = np.zeros_like(temp_map)
    valid_mask = temp_map >= 2273.15
    evap_map[valid_mask] = A * np.exp(B / temp_map[valid_mask])
    return evap_map

def calculate_lifespan_map(binary_grid, evap_map):
    edt_dist = ndimage.distance_transform_edt(binary_grid)
    L = edt_dist * VOXEL_SIZE
    valid_mask = (binary_grid > 0.5)
    safe_evap = np.maximum(evap_map, 1e-15)
    lifespan_map = np.zeros_like(evap_map)
    lifespan_map[valid_mask] = (0.1 * edt_dist[valid_mask] * VOXEL_SIZE * RHO_DENSITY) / safe_evap[valid_mask]
    lifespan_map[~valid_mask] = 0.0
    return lifespan_map

def extract_mesh(self):
    binary_grid = self.current_voxel_grid
    verts, faces, _, _ = measure.marching_cubes(binary_grid, level=0.5)
    mesh = trimesh.Trimesh(vertices=verts * self.voxel_size, faces=faces)
    
    # 保留最大连通域
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        areas = [c.area for c in components]
        mesh = components[np.argmax(areas)]

    trimesh.smoothing.filter_laplacian(mesh, iterations=5) # 平滑网格
    
    # 填补平滑后网格可能产生的孔洞
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)

    return mesh