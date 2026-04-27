# 工具库
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN
from scipy.interpolate import griddata
import os
import mph

# 全局配置参数
GRID_SIZE = (100, 100, 150)
VOXEL_SIZE = 0.1       
RHO_DENSITY = 19350.0  
CORES = 16

def initialize_cylinder_grid():
    nx, ny, nz = GRID_SIZE
    grid = np.zeros((nx, ny, nz), dtype=np.float32)
    radius = 25  
    x, y, z = np.ogrid[:nx, :ny, :nz]
    dist_sq = (x - nx//2)**2 + (y - ny//2)**2
    cylinder_mask = (dist_sq <= radius**2) & (z >= 0) & (z <= nz)
    grid[cylinder_mask] = 1.0
    return grid

def extract_characteristic_length(voxel_grid):
    # 提取三维骨架特征长度
    edt_distances = distance_transform_edt(voxel_grid)
    skeleton = skeletonize(voxel_grid)
    skeleton_indices = np.where(skeleton > 0)
    local_thickness = 2.0 * edt_distances[skeleton_indices]

    if len(local_thickness) == 0:
        return {}, np.array([]), np.array([])
        
    coords = np.column_stack(skeleton_indices)
    dbscan = DBSCAN(eps=1.74, min_samples=2)
    cluster_labels = dbscan.fit_predict(coords)
    
    feature_clusters = {}
    for cluster_id in set(cluster_labels):
        if cluster_id == -1: continue
        cluster_mask = (cluster_labels == cluster_id)
        feature_clusters[cluster_id] = np.mean(local_thickness[cluster_mask])
                
    return feature_clusters, coords, cluster_labels

def calculate_evaporation_rate(temp_map):
    # 计算蒸发率图谱
    safe_temp = np.maximum(temp_map, 2273.15) # 2000℃以上考虑蒸发
    A = 3.9e7       
    B = -1.023e5    
    evap_map = A * np.exp(B / safe_temp)
    return evap_map

def calculate_structural_lifespan(binary_grid, evap_map):
    # 寿命计算

    if np.sum(binary_grid) < 10:
        return 0.0
        
    # 提取内部距离场 (单位: voxel)
    edt_dist = distance_transform_edt(binary_grid)
    
    # 特征长度  L = EDT * VOXEL_SIZE (转为 m)
    L = edt_dist * (VOXEL_SIZE / 1000.0)
    
    # 只计算实体
    valid_mask = (binary_grid > 0.5)
    
    # 寿命图谱 t_fail = 0.2 * L * rho / v_e
    # 公式化简: t_fail = 0.1 * EDT_voxel * VOXEL_SIZE_m * RHO / v_e
    safe_evap = np.maximum(evap_map, 1e-15)
    lifespan_map = np.zeros_like(evap_map)
    lifespan_map[valid_mask] = (0.2 * edt_dist[valid_mask] * (VOXEL_SIZE / 1000.0) * RHO_DENSITY) / safe_evap[valid_mask]
    
    # 全局最薄弱点决定了整体寿命
    min_lifespan = np.min(lifespan_map[valid_mask])
    return min_lifespan

def Topo_Init_evaluation(client):
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_path, 'Topo_Init_3D.mph')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到基准模型: {model_path}")

    model = client.load(model_path)
    
    mesh_tags = model.java.component("comp1").mesh().tags()
    if len(mesh_tags) > 0:
        model.java.component("comp1").mesh(str(mesh_tags[0])).run()
        
    study_tags = model.java.study().tags()
    model.java.study(str(study_tags[0])).run()
    
    # 提取全局物理量
    net_rad = float(model.evaluate('Total_Rad'))
    p_in = float(model.evaluate('P_in'))
    
    # 提取空间温度场并插值到网格
    x_coords = model.evaluate('x') * 1000
    y_coords = model.evaluate('y') * 1000
    z_coords = model.evaluate('z') * 1000
    temp_points = model.evaluate('T')
    
    nx, ny, nz = GRID_SIZE
    x_g, y_g, z_g = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    temp_map = griddata(
        (x_coords.ravel(), y_coords.ravel(), z_coords.ravel()), 
        temp_points.ravel(), 
        (x_g * VOXEL_SIZE, y_g * VOXEL_SIZE, z_g * VOXEL_SIZE), 
        method='nearest'
    ).astype(np.float32)
    temp_map = np.nan_to_num(temp_map, nan=300.0)
    
    grid_3d = initialize_cylinder_grid()
    evap_map = calculate_evaporation_rate(temp_map)
    
    # 计算寿命
    initial_life = calculate_structural_lifespan(grid_3d, evap_map)
    initial_efficiency = (net_rad / p_in) if p_in > 1e-5 else 0.0

    #清理模型
    model.clear() 

    return initial_life, net_rad, initial_efficiency