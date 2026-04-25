import numpy as np
import mph
import os
from scipy.interpolate import griddata

# 核心导入：直接引入我们刚刚写的共享工具
from physics_tools import (
    GRID_SIZE, VOXEL_SIZE, 
    initialize_cylinder_grid, 
    extract_characteristic_length, 
    calculate_evaporation_rate, 
    calculate_structural_lifespan
)
CORES = 16


def topo_init_evaluation():
    # 初始拓扑评估
    # COMSOL通信
    print("正在启动 COMSOL 服务器...")
    current_path = os.path.dirname(os.path.abspath(__file__))
    client = mph.start(cores=CORES)
    model_path = os.path.join(current_path, 'Topo_Init_3D.mph')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
        
    model = client.load(model_path)
    
    # COMSOL仿真
    mesh_tags = model.java.component("comp1").mesh().tags()
    if len(mesh_tags) > 0:
        target_mesh = str(mesh_tags[0])
        model.java.component("comp1").mesh(target_mesh).run()
        
    study_tags = model.java.study().tags()
    if len(study_tags) == 0:
        raise ValueError("模型中未找到研究(Study)节点！")
    target_study = str(study_tags[0])
    model.java.study(target_study).run()
    
    # 提取全局物理量
    net_rad = float(model.evaluate('Total_Rad'))
    p_in = float(model.evaluate('P_in'))
    max_temp = float(model.evaluate('Max_Temp'))
    
    # 提取空间温度场，COMSOL返回的单位为米
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
    
    # 计算寿命评估指标
    grid_3d = initialize_cylinder_grid() 
    evap_map = calculate_evaporation_rate(temp_map)
    feature_clusters, coords, cluster_labels = extract_characteristic_length(grid_3d)
    
    # 4. 打印最终基准报告
    print("\n" + "="*50)
    print("📊 内置初始结构物理性能基准报告 📊")
    print("="*50)
    print(f"🌡️ 最高温度 (Max Temp)     : {max_temp:.2f} K")
    print(f"⚡ 输入电功率 (P_in)        : {p_in:.2f} W")
    print(f"🌞 净辐射功率 (Total Rad)  : {net_rad:.2f} W")
    print(f"🎯 能量转换效率 (Efficiency): {efficiency * 100:.4f} %")
    print(f"⏳ 最短有效寿命 (Lifespan)   : {initial_life:.2e} 秒")
    print("="*50)

if __name__ == "__main__":
    run_Topo_Init_evaluation()