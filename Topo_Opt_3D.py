import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mph
import os
import trimesh
import scipy.ndimage as ndimage
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
from skimage import measure
from stable_baselines3 import PPO
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from physics_tools import (
    GRID_SIZE, 
    VOXEL_SIZE, 
    RHO_DENSITY,
    initialize_cylinder_grid, 
    extract_characteristic_length, 
    calculate_evaporation_rate, 
    calculate_structural_lifespan,
    Topo_Init_evaluation
    )


# 限制PyTorch使用的CPU线程数，把更多算力留给后端的 COMSOL 物理场求解
torch.set_num_threads(16)

class TungstenTopologyEnv(gym.Env):
    def __init__(self):
        super(TungstenTopologyEnv, self).__init__()
        
        # 定义全局参数
        self.grid_size = GRID_SIZE
        self.voxel_size = VOXEL_SIZE
        self.rho_density = RHO_DENSITY
        
        # 初始化通信
        print("正在启动 COMSOL 服务器...")
        self.client = mph.start(cores=16)
        
        current_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_path,'Topo_Opt_3D.mph')
        if os.path.exists(model_path):
            self.model = self.client.load(model_path)
        else:
            print(f"【警告】未找到 {model_path}，使用随机假数据模式供调试。")
            self.model = None
            
        # 定义状态空间与动作空间
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(3, self.grid_size[0], self.grid_size[1], self.grid_size[2]), 
            dtype=np.float32
        )
        # 3D 动作空间需 6 个参数：(x1, y1, z1) 用于挖除，(x2, y2, z2) 用于填补
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # 初始化环境演化参数
        self.step_count = 0
        self.output_dir = "./topology_evolution"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 生成带电极掩码的初始体素网格
        self.current_voxel_grid = self._initialize_agent_grid()

        # 获取初始参数。
        print("正在进行初始物理基准评估，请稍候...")
        try:
            init_life, init_rad, init_eff = Topo_Init_evaluation()
            self.initial_life = init_life
            self.initial_radiation = init_rad
            self.initial_efficiency = init_eff
            print(f"【基准获取成功】寿命: {init_life:.2e}s | 辐射: {init_rad:.2f}W | 效率: {init_eff*100:.2f}%")
        except Exception as e:
            print(f"【警告】基准提取失败，使用安全默认值。错误: {e}")
            self.initial_life = 100.0
            self.initial_radiation = 50.0
            self.initial_efficiency = 0.5

    def _initialize_agent_grid(self):
        # 初始化智能体视角的网格。
        # 添加“不可修改电极区域”。
        grid = initialize_cylinder_grid()
        
        nx, ny, nz = self.grid_size
        self.frozen_mask = np.zeros_like(grid, dtype=bool)
        radius = 25 
        z_start, z_end = 0, nz
        
        x, y, z = np.ogrid[:nx, :ny, :nz]
        dist_sq = (x - nx//2)**2 + (y - ny//2)**2
        
        electrode_thickness = 2 # 电极厚度为2个体素
        bottom_electrode = (dist_sq <= radius**2) & (z >= z_start) & (z < z_start + electrode_thickness)
        top_electrode = (dist_sq <= radius**2) & (z > z_end - electrode_thickness) & (z <= z_end)
        
        self.frozen_mask[bottom_electrode] = True
        self.frozen_mask[top_electrode] = True
        
        return grid

    def _run_comsol_simulation_3d(self, grid_3d):
        #运行 COMSOL 联合仿真
        nx, ny, nz = self.grid_size
        
        if self.model is None:
            return np.ones((nx, ny, nz), dtype=np.float32) * 2500.0, 50.0, 100.0, 2500.0

        closed_grid = ndimage.binary_closing(grid_3d, structure=np.ones((3, 3, 3))).astype(np.float32)
        smoothed_grid = ndimage.gaussian_filter(closed_grid, sigma=1.5)
        
        try:
            verts, faces, normals, values = measure.marching_cubes(smoothed_grid, 0.5)
            mesh = trimesh.Trimesh(vertices=verts * self.voxel_size, faces=faces)
            mesh.fill_holes() 
            trimesh.smoothing.filter_laplacian(mesh, iterations=5) 
            
            current_path = os.path.dirname(os.path.abspath(__file__))
            stl_path = os.path.join(current_path, 'temp_topology.stl')
            mesh.export(stl_path)

            geom = self.model.java.component("comp1").geom("geom1")
            geom.feature("imp1").set("filename", stl_path)
            geom.run()

            mesh_tags = self.model.java.component("comp1").mesh().tags()
            target_mesh = str(mesh_tags[0]) if len(mesh_tags) > 0 else "mesh1"
            if len(mesh_tags) == 0: self.model.java.component("comp1").mesh().create(target_mesh)
            self.model.java.component("comp1").mesh(target_mesh).run()
            
            study_tags = self.model.java.study().tags()
            target_study = str(study_tags[0])
            self.model.java.study(target_study).run()

            net_rad = self.model.evaluate('Total_Rad')       
            p_in = self.model.evaluate('P_in')           
            max_temp = self.model.evaluate('Max_Temp') 
            
            x_coords = self.model.evaluate('x') * 1000  
            y_coords = self.model.evaluate('y') * 1000
            z_coords = self.model.evaluate('z') * 1000
            temp_points = self.model.evaluate('T')

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
            
            return temp_map, float(net_rad), float(p_in), float(max_temp)
            
        except Exception as e:
            print(f"【拦截】COMSOL 几何重构或求解失败，错误信息: {str(e)[:100]}...")
            raise RuntimeError("COMSOL Simulation Failed")

    def _check_structural_integrity(self):
        # 连通性检查
        structure = ndimage.generate_binary_structure(3, 3) 
        labeled_array, num_features = ndimage.label(self.current_voxel_grid, structure=structure)
        if num_features > 1:
            return False
        return True

    def step(self, action):
        self.step_count += 1
        
        # 拓扑优化动作
        success, modified_count = self._apply_3d_brush_action(action, brush_radius=3)
        if not success:
            return self._get_obs(), -2.0, False, False, {"error": "Invalid action"}
            
        # 物理约束：拦截孤立碎片
        if not self._check_structural_integrity():
            return self._get_obs(), -50.0, True, False, {"error": "Structural fragmentation"}
            
        # 执行 COMSOL 联合仿真
        try:
            temp_map, net_rad, p_in, max_temp = self._run_comsol_simulation_3d(self.current_voxel_grid)
        except Exception as e:
            return self._get_obs(), -100.0, True, False, {"error": "COMSOL Non-convergence"}
            
        # 温度约束
        if max_temp > 3273.15:
            return self._get_obs(), -50.0, True, False, {"error": "Temperature limit exceeded"}
            
        # 性能评估
        evap_map = calculate_evaporation_rate(temp_map) 
        feature_clusters, coords, cluster_labels = extract_characteristic_length(self.current_voxel_grid)
        
        if not feature_clusters:
            return self._get_obs(), -100.0, True, False, {"error": "No solid structure left"}

        # 寿命评估
        topo_life, _ = calculate_structural_lifespan(feature_clusters, evap_map, coords, cluster_labels)

        efficiency = net_rad / p_in if p_in > 1e-5 else 0.0 

        self._save_evolution_frame(efficiency, topo_life)
        reward = self._calculate_reward(efficiency, topo_life)
        obs = self._get_obs(temp_map, evap_map)
        
        return obs, reward, False, False, {
            "life": topo_life, 
            "radiation": net_rad,
            "efficiency": efficiency,
            "max_temp": max_temp
        }
        
    def _calculate_reward(self, efficiency, topo_life):
        # 计算奖励
        w1 = 20.0 # 辐射功率奖励系数
        r_eff = w1 * np.clip((efficiency / self.initial_efficiency), -5.0, 5.0)
        
        w2 = 5.0 # 拓扑寿命奖励系数
        p_life = 0.0
        red_line = 0.3 * self.initial_life
        
        if topo_life < red_line:
            shortfall_ratio = (red_line - topo_life) / red_line
            p_life = -50.0 * (shortfall_ratio ** 2)
        else:
            p_life = w2 * np.clip((topo_life / self.initial_life), 0.0, 2.0)
            
        return float(r_eff + p_life)

    def _save_evolution_frame(self, efficiency, topo_life):
        # 保存演化帧和指标日志
        if self.step_count % 10 != 0:
            return
            
        smoothed_grid = ndimage.gaussian_filter(self.current_voxel_grid, sigma=1.0)
        verts, faces, normals, values = measure.marching_cubes(smoothed_grid, 0.5)
        mesh = trimesh.Trimesh(vertices=verts * self.voxel_size, faces=faces)
        
        stl_filename = os.path.join(self.output_dir, f"frame_{self.step_count:05d}.stl")
        mesh.export(stl_filename)
        
        log_path = os.path.join(self.output_dir, "performance_log.csv")
        write_header = not os.path.exists(log_path)
        with open(log_path, 'a') as f:
            if write_header: f.write("Step,Efficiency,LifeSpan\n")
            f.write(f"{self.step_count},{efficiency:.4f},{topo_life:.2f}\n")

    def reset(self, seed=None):
        # 回合重置
        super().reset(seed=seed)
        self.current_voxel_grid = self._initialize_agent_grid()
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

        # 取两者最小数量
        for i in range(swap_count):
            self.current_voxel_grid[rm_candidates[i]] = 0.0
            self.current_voxel_grid[add_candidates[i]] = 1.0

        return True, swap_count

    def _get_obs(self, temp=None, evap=None):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        inside_dist = distance_transform_edt(self.current_voxel_grid)
        outside_dist = distance_transform_edt(1 - self.current_voxel_grid)
        sdf = inside_dist - outside_dist
        obs[0] = np.clip(sdf / 50.0, -1.0, 1.0)
        if temp is not None: obs[1] = np.clip(temp / 3600.0, 0.0, 1.0)
        if evap is not None:
            safe_evap = np.maximum(evap, 1e-15) 
            obs[2] = np.clip((np.log10(safe_evap) + 15.0) / 10.0, 0.0, 1.0)
        return obs

class Custom3DCNN(BaseFeaturesExtractor):
    # 定义三维神经网络
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(Custom3DCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv3d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), 
            
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), 
            
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Flatten()
        )
        
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# 运行主程序
if __name__ == "__main__":
    env = TungstenTopologyEnv()
    
    policy_kwargs = dict(
        features_extractor_class=Custom3DCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False 
    )
    
    model = PPO(
        "CnnPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        n_steps=512,
        batch_size=32, 
        learning_rate=3e-4,
        n_epochs=10, 
        verbose=1, 
        tensorboard_log="./tungsten_ppo_tensorboard/"
    )
    
    print("开始强化学习训练...")
    model.learn(total_timesteps=10000)
    model.save("ppo_tungsten_3D_optimized")