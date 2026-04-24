import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mph
import os
import trimesh
import scipy.ndimage as ndimage
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage import measure
from stable_baselines3 import PPO
from sklearn.cluster import DBSCAN
import jpype
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 限制PyTorch使用的CPU线程数，把更多算力留给后端的 COMSOL 物理场求解
torch.set_num_threads(4)

class TungstenTopologyEnv(gym.Env):
    def __init__(self, grid_size=(100, 100, 150)):
        super(TungstenTopologyEnv, self).__init__()
        self.grid_size = grid_size
        
        # 1 初始化 COMSOL 通信
        print("正在启动 COMSOL 后台服务器...")
        self.client = mph.start(cores=24)
        
        # 加载基准模型文件
        current_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_path,'Topo_Opt_3D.mph')
        if os.path.exists(model_path):
            self.model = self.client.load(model_path)
        else:
            print(f"【警告】未找到 {model_path}，使用随机假数据模式供调试。")
            self.model = None
            
        self.voxel_size = 0.1 # 每个体素代表 0.1mm 的实际尺寸
        
        # 2 状态空间与动作空间 (3D 空间)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(3, grid_size[0], grid_size[1], grid_size[2]), 
            dtype=np.float32
        )
        # 3D 动作空间需 6 个参数：(x1, y1, z1) 用于挖除，(x2, y2, z2) 用于填补
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # 3 初始化参数
        self.initial_radiation = 1.0 
        self.initial_life = 100.0
        self.rho_density = 19350.0
        self.current_voxel_grid = self._initialize_cylinder()

    def _initialize_cylinder(self):
        nx, ny, nz = self.grid_size
        grid = np.zeros((nx, ny, nz), dtype=np.float32)
        radius = 25 
        z_start = 0
        z_end = nz
        
        x, y, z = np.ogrid[:nx, :ny, :nz]
        dist_sq = (x - nx//2)**2 + (y - ny//2)**2
        
        # 生成实体
        cylinder_mask = (dist_sq <= radius**2) & (z >= z_start) & (z <= z_end)       
        grid[cylinder_mask] = 1.0       
        
        self.frozen_mask = np.zeros_like(grid, dtype=bool)
        electrode_thickness = 2 # 电极厚度为2个体素
        bottom_electrode = (dist_sq <= radius**2) & (z >= z_start) & (z < z_start + electrode_thickness)
        top_electrode = (dist_sq <= radius**2) & (z > z_end - electrode_thickness) & (z <= z_end)
        self.frozen_mask[bottom_electrode] = True
        self.frozen_mask[top_electrode] = True
        
        return grid

    def extract_characteristic_length(self, voxel_grid):
        edt_distances = distance_transform_edt(voxel_grid)  
        skeleton = skeletonize(voxel_grid)  
        skeleton_indices = np.where(skeleton > 0)
        local_thickness = 2.0 * edt_distances[skeleton_indices] 

        if len(local_thickness) == 0: 
            return {}, np.array([]), np.array([]) 
        
        coords = np.column_stack(skeleton_indices)  
        dbscan = DBSCAN(eps=1.74, min_samples=2) # 3D中2体素的连通对角距离约为 1.732
        cluster_labels = dbscan.fit_predict(coords)
        
        feature_clusters = {} 
        unique_labels = set(cluster_labels)
        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue
            cluster_mask = (cluster_labels == cluster_id)
            cluster_thickness = local_thickness[cluster_mask]
            feature_clusters[cluster_id] = np.mean(cluster_thickness)
                    
        return feature_clusters, coords, cluster_labels 

    def _calculate_evaporation_rate(self, temp_map):
        safe_temp = np.maximum(temp_map, 2000.0) 
        A = 3.9e7 # 单位由g/(cm²·s)换算为 kg/（㎡·s)    
        B = -1.023e5  
        evap_map = A * np.exp(B / safe_temp) 
        return evap_map

    # COMSOL 联合仿真
    def _run_comsol_simulation_3d(self, grid_3d):
        nx, ny, nz = self.grid_size
        
        if self.model is None:
            return np.ones((nx, ny, nz), dtype=np.float32) * 2500.0, 50.0

        # 1. 3D平滑与 STL 生成
        smoothed_grid = ndimage.gaussian_filter(grid_3d, sigma=1.0)
        try:
            verts, faces, normals, values = measure.marching_cubes(smoothed_grid, 0.5)
            
            # 使用 trimesh 导出真实的物理尺寸 STL
            mesh = trimesh.Trimesh(vertices=verts * self.voxel_size, faces=faces)
            current_path = os.path.dirname(os.path.abspath(__file__))
            stl_path = os.path.join(current_path, 'temp_topology.stl')
            mesh.export(stl_path)
            
            # 2. COMSOL 几何更新
            geom = self.model.java.component("comp1").geom("geom1")
            # 确保 COMSOL 基准模型里预先建立了一个 Import 节点，Tag 为 imp1
            try:
                geom.feature("imp1").set("filename", stl_path)
            except Exception as e:
                raise ValueError("COMSOL 模型中未找到名为 'imp1' 的导入节点！请在模型中手动添加几何导入。")
            geom.run()

            # 3. 动态网格操作
            mesh_tags = self.model.java.component("comp1").mesh().tags()
            if len(mesh_tags) == 0:
                self.model.java.component("comp1").mesh().create("mesh1")
                target_mesh = "mesh1"
            else:
                target_mesh = str(mesh_tags[0])
            self.model.java.component("comp1").mesh(target_mesh).run()
            
            # 4. 求解操作
            study_tags = self.model.java.study().tags()
            if len(study_tags) == 0:
                raise ValueError("模型中丢失了研究节点！")
            target_study = str(study_tags[0])
            self.model.java.study(target_study).run()

            # 5. 提取物理场数据
            net_rad = self.model.evaluate('Total_Rad')
            x_coords = self.model.evaluate('x') * 1000  # 转换为毫米单位
            y_coords = self.model.evaluate('y') * 1000
            z_coords = self.model.evaluate('z') * 1000
            temp_points = self.model.evaluate('T')

            # 6. 三维网格化插值
            x_grid, y_grid, z_grid = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
            x_grid_m = x_grid * self.voxel_size
            y_grid_m = y_grid * self.voxel_size
            z_grid_m = z_grid * self.voxel_size
            
            temp_map = griddata(
                (x_coords.ravel(), y_coords.ravel(), z_coords.ravel()), 
                temp_points.ravel(), 
                (x_grid_m, y_grid_m, z_grid_m), 
                method='nearest'
            ).astype(np.float32)
            
            temp_map = np.nan_to_num(temp_map, nan=300.0)
            
            return temp_map, float(net_rad)
            
        except Exception as e:
            print(f"【拦截】显式几何重构或求解失败: {e}")
            raise e  

    def step(self, action):
        success, modified_count = self._apply_3d_brush_action(action, brush_radius=3)
        if not success:
            return self._get_obs(), -2.0, False, False, {"error": "Invalid action"}
            
        try:
            temp_map, net_rad = self._run_comsol_simulation_3d(self.current_voxel_grid)
        except Exception as e:
            return self._get_obs(), -100.0, True, False, {"error": "COMSOL Non-convergence"}
            
        evap_map = self._calculate_evaporation_rate(temp_map) 
        feature_clusters, coords, cluster_labels = self.extract_characteristic_length(self.current_voxel_grid)
        
        if not feature_clusters:
            return self._get_obs(), -100.0, True, False, {"error": "No solid structure left"}

        cluster_lifespans = []
        for cluster_id, L_k0 in feature_clusters.items():
            v_e_k = self._get_mapped_evaporation(evap_map, coords, cluster_labels, cluster_id)
            L_real = L_k0 * self.voxel_size
            t_fail_k = (0.2 * L_k0 * self.rho_density) / v_e_k
            cluster_lifespans.append(t_fail_k)
            
        topo_life = np.min(cluster_lifespans)

        reward = self._calculate_reward(net_rad, topo_life)
        obs = self._get_obs(temp_map, evap_map)
        
        return obs, reward, False, False, {"life": topo_life, "radiation": net_rad}
        
    def _calculate_reward(self, net_rad, topo_life):
        w1 = 10.0
        r_rad = w1 * np.clip((net_rad / self.initial_radiation), -5.0, 5.0)
        w2 = 1.0
        p_life = 0.0
        red_line = 0.3 * self.initial_life
        if topo_life < red_line:
            shortfall_ratio = (red_line - topo_life) / red_line
            p_life = -50.0 * (shortfall_ratio ** 2)
            
        return float(r_rad + w2 * p_life)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_voxel_grid = self._initialize_cylinder()
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}
        
    def _apply_3d_brush_action(self, action, brush_radius=2):
        # 3D 绝对体积守恒球形笔刷
        nx, ny, nz = self.grid_size
        
        x1 = int(np.clip((action[0] + 1.0) / 2.0 * (nx - 1), 0, nx - 1))
        y1 = int(np.clip((action[1] + 1.0) / 2.0 * (ny - 1), 0, ny - 1))
        z1 = int(np.clip((action[2] + 1.0) / 2.0 * (nz - 1), 0, nz - 1))
        
        x2 = int(np.clip((action[3] + 1.0) / 2.0 * (nx - 1), 0, nx - 1))
        y2 = int(np.clip((action[4] + 1.0) / 2.0 * (ny - 1), 0, ny - 1))
        z2 = int(np.clip((action[5] + 1.0) / 2.0 * (nz - 1), 0, nz - 1))

        rm_candidates, add_candidates = [], []
        
        for dx in range(-brush_radius, brush_radius + 1):
            for dy in range(-brush_radius, brush_radius + 1):
                for dz in range(-brush_radius, brush_radius + 1):
                    if dx**2 + dy**2 + dz**2 <= brush_radius**2:
                        # 收集待挖点
                        cx1, cy1, cz1 = x1 + dx, y1 + dy, z1 + dz
                        if 0 <= cx1 < nx and 0 <= cy1 < ny and 0 <= cz1 < nz:
                            if self.current_voxel_grid[cx1, cy1, cz1] == 1.0 and not self.frozen_mask[cx1, cy1, cz1]:
                                rm_candidates.append((cx1, cy1, cz1))
                        # 收集待填点
                        cx2, cy2, cz2 = x2 + dx, y2 + dy, z2 + dz
                        if 0 <= cx2 < nx and 0 <= cy2 < ny and 0 <= cz2 < nz:
                            if self.current_voxel_grid[cx2, cy2, cz2] == 0.0 and not self.frozen_mask[cx2, cy2, cz2]:
                                add_candidates.append((cx2, cy2, cz2))

        swap_count = min(len(rm_candidates), len(add_candidates))
        if swap_count == 0:
            return False, 0

        # 取两者最小数量，严格保证总质量 1:1 无损对换
        for i in range(swap_count):
            self.current_voxel_grid[rm_candidates[i]] = 0.0
            self.current_voxel_grid[add_candidates[i]] = 1.0

        return True, swap_count

    def _get_mapped_evaporation(self, evap_map, coords, cluster_labels, cluster_id):
        mask = (cluster_labels == cluster_id)
        cluster_coords = coords[mask]
        evap_values = []
        for coord in cluster_coords:
            x, y, z = coord[0], coord[1], coord[2]
            evap_values.append(evap_map[x, y, z])
            
        if len(evap_values) == 0: return 1e-10
        return np.max(evap_values) 

    def _get_obs(self, temp=None, evap=None):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        inside_dist = distance_transform_edt(self.current_voxel_grid)
        outside_dist = distance_transform_edt(1 - self.current_voxel_grid)
        sdf = inside_dist - outside_dist
        obs[0] = np.clip(sdf / 50.0, -1.0, 1.0)
        if temp is not None:
            obs[1] = np.clip(temp / 3600.0, 0.0, 1.0)
        if evap is not None:
            safe_evap = np.maximum(evap, 1e-15) 
            log_evap = np.log10(safe_evap)
            obs[2] = np.clip((log_evap + 15.0) / 10.0, 0.0, 1.0)
        return obs

# 定义3D CNN网络
class Custom3DCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(Custom3DCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        # 加入 MaxPool3d 快速降采样
        self.cnn = nn.Sequential(
            # 第一层：提取初始特征，32通道，并立刻将尺寸缩小一半
            nn.Conv3d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), 
            
            # 第二层：64通道，尺寸再缩小一半
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), 
            
            # 第三层：128通道，尺寸再缩小一半
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Flatten()
        )
        
        # 自动推导 Flatten 后的神经元数量
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

if __name__ == "__main__":
    env = TungstenTopologyEnv()
    
    # 注入我们自定义的 3D CNN
    policy_kwargs = dict(
        features_extractor_class=Custom3DCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False # 禁用默认的 /255 像素归一化
    )
    
    model = PPO(
        "CnnPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        n_steps=512, # 如果显存爆炸则将参数调小 
        batch_size=32, 
        learning_rate=3e-4,
        n_epochs=10, 
        verbose=1, 
        tensorboard_log="./tungsten_ppo_tensorboard/"
    )
    
    print("开始强化学习三维训练...")
    model.learn(total_timesteps=10000)
    model.save("ppo_tungsten_3D_optimized")