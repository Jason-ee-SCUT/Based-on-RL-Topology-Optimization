# 工具箱
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN
from scipy.interpolate import griddata

# 全局配置参数
GRID_SIZE = (100, 100, 150)
VOXEL_SIZE = 0.1       
RHO_DENSITY = 19350.0  

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
    safe_temp = np.maximum(temp_map, 2000.0)
    A = 3.9e7       
    B = -1.023e5    
    evap_map = A * np.exp(B / safe_temp)
    return evap_map

def calculate_structural_lifespan(feature_clusters, evap_map, coords, cluster_labels):
    # 计算寿命，返回最短寿命和所有聚类的寿命列表
    if not feature_clusters:
        return 0.0, []
        
    lifespans = []
    for cluster_id, L_k0 in feature_clusters.items():
        # 提取当前特征块的最大蒸发率
        mask = (cluster_labels == cluster_id)
        evap_vals = [evap_map[c[0], c[1], c[2]] for c in coords[mask]]
        v_e_k = np.max(evap_vals) if evap_vals else 1e-10
        
        # 特征长度转换为米(m)并计算失效时间
        L_meter = L_k0 * VOXEL_SIZE / 1000.0 
        t_fail_k = (0.2 * L_meter * RHO_DENSITY) / v_e_k
        lifespans.append(t_fail_k)
        
    return np.min(lifespans), lifespans

def get_initial_baseline_metrics(model, grid_3d):
    # 静默求解 COMSOL 初始拓扑，提取并返回基准性能
    current_path = os.path.dirname(os.path.abspath(__file__))
    client = mph.start(cores=CORES)
    model_path = os.path.join(current_path, 'Topo_Init_3D.mph')
    # 运行现存网格与研究
    mesh_tags = model.java.component("comp1").mesh().tags()
    if len(mesh_tags) > 0:
        target_mesh = str(mesh_tags[0])
        model.java.component("comp1").mesh(target_mesh).run()
        
    study_tags = model.java.study().tags()
    target_study = str(study_tags[0])
    model.java.study(target_study).run()
    
    # 提取全局物理量
    net_rad = float(model.evaluate('Total_Rad'))
    p_in = float(model.evaluate('P_in'))
    
    # 提取空间温度场并插值
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
    
    # 计算寿命与效率
    evap_map = calculate_evaporation_rate(temp_map)
    feature_clusters, coords, cluster_labels = extract_characteristic_length(grid_3d)
    initial_life, _ = calculate_structural_lifespan(feature_clusters, evap_map, coords, cluster_labels)
    efficiency = (net_rad / p_in) if p_in > 1e-5 else 0.0

    # 返回初始拓扑性能
    return initial_life, net_rad, efficiency