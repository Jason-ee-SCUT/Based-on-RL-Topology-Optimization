# 工具库
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN
from scipy.interpolate import griddata
import os
import mph
import trimesh
import networkx as nx
from skimage import measure

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

def extract_optimized_mesh(density_field, voxel_size, level=0.5):
    """
    从密度场提取优化的钨加热部件网格。
    包含：等值面提取、物理坐标转换、退化面过滤、智能小孔填充、法线修复。
    
    参数:
        density_field: 3D numpy array, 归一化密度场 (0-1)
        voxel_size: float, 体素物理尺寸 (米), 例如 1e-4 (0.1mm)
        level: float, Marching Cubes 等值面阈值
    
    返回:
        trimesh.Trimesh: 修复后的网格对象
    """
    # 1提取等值面
    try:
        verts, faces, _, _ = measure.marching_cubes(density_field, level=level)
    except ValueError:
        return trimesh.Trimesh()

    if len(verts) == 0 or len(faces) == 0:
        return trimesh.Trimesh()

    # 转换为物理坐标
    mesh = trimesh.Trimesh(vertices=verts * voxel_size, faces=faces)
    
    # 2合并重合顶点
    mesh.merge_vertices()

    # 3面积过滤：移除退化面
    face_areas = mesh.area_faces
    if len(face_areas) > 0:
        avg_area = mesh.area / len(face_areas)
        # 阈值：相对阈值 (平均面积 * 1e-4) 与 绝对阈值 (1e-10 m^2) 取大者
        threshold = max(1e-10, avg_area * 1e-4) 
        valid_mask = face_areas > threshold
    
        if not np.all(valid_mask):
            mesh.update_faces(valid_mask)
            mesh.remove_unreferenced_vertices()
        
        if len(mesh.faces) < 4:
            return mesh

    # 4填充微小孔洞
    # 策略：仅填充直径 < 2.5 * voxel_size 的孔洞 (视为数值噪声)
    max_hole_diameter = voxel_size * 2.5  
    
    boundary_edges = mesh.boundary_edges
    
    if len(boundary_edges) > 0:
        g = nx.Graph()
        g.add_edges_from(boundary_edges)
    
        holes_to_fill = []
    
        for component in nx.connected_components(g):
            boundary_verts_idx = list(component)
            if len(boundary_verts_idx) < 3:
                continue
            
            hole_verts = mesh.vertices[boundary_verts_idx]
            
            # 估算孔洞大小 (包围盒对角线)
            span = np.max(hole_verts, axis=0) - np.min(hole_verts, axis=0)
            diameter_est = np.linalg.norm(span)
            
            if diameter_est < max_hole_diameter:
                # 收集属于该组件的边
                component_edges = []
                # 优化查找效率：先建立集合
                comp_set = set(component)
                for u, v in mesh.boundary_edges:
                    if u in comp_set and v in comp_set:
                        component_edges.append((u, v))
                
                if len(component_edges) >= 3:
                    holes_to_fill.append(component_edges)

        # 执行填充
        for edge_loop in holes_to_fill:
            try:
                mesh = trimesh.holes.fill_hole(mesh, edge_loop)
            except Exception:
                continue

    # 5二次清理
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) > 0:
        face_areas = mesh.area_faces
        if len(face_areas) > 0:
            valid_mask = face_areas > (mesh.area / len(mesh.faces) * 1e-6)
            if not np.all(valid_mask):
                mesh.update_faces(valid_mask)
                mesh.remove_unreferenced_vertices()

    # 6修复法线
    if len(mesh.faces) > 0:
        mesh.fix_normals()
        
        # 7轻微平滑 (仅当网格足够复杂)
        if len(mesh.faces) > 15:
            try:
                trimesh.smoothing.filter_laplacian(mesh, iterations=1)
            except Exception:
                pass

    return mesh