import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mph
import os
import trimesh
import scipy.ndimage as ndimage
from scipy.interpolate import griddata
from skimage import measure
from stable_baselines3 import PPO
import torch
import torch.nn as nn
import cupy as cp
from cupyx.scipy.ndimage import distance_transform_edt as c_edt
from cupyx.scipy.ndimage import gaussian_filter as c_gaussian
from cupyx.scipy.ndimage import zoom as c_zoom
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from physics_tools import (
    GRID_SIZE, VOXEL_SIZE, RHO_DENSITY,
    initialize_cylinder_grid, calculate_evaporation_rate, 
     Topo_Init_evaluation
)

class TungstenTopologyEnv(gym.Env):
    def __init__(self, client):
        super(TungstenTopologyEnv, self).__init__()
        self.client = client
        self.grid_size = GRID_SIZE
        self.voxel_size = VOXEL_SIZE
        self.rho_density = RHO_DENSITY
        
        self.max_steps = 1000 
        
        current_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_path, 'Topo_Opt_3D.mph')
        self.model = self.client.load(model_path) if os.path.exists(model_path) else None
            
        # 状态空间为3通道: [SDF边界, 温度场, 寿命场]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(3, 50, 50, 75),
            dtype=np.float32
            )
        
        # 动作空间：低频连续密度修改
        self.action_shape = (self.grid_size[0]//5, self.grid_size[1]//5, self.grid_size[2]//5)
        self.action_space = spaces.Box(low=-0.15, high=0.15, shape=self.action_shape, dtype=np.float32)
        
        self.step_count = 0
        self.output_dir = "./topology_evolution"
        os.makedirs(self.output_dir, exist_ok=True)
        self.target_volume = 0.0
        self._initialize_masks()

        # 基准评估
        try:
            init_life, init_rad, init_eff = Topo_Init_evaluation(self.client)
            self.initial_life = init_life
            self.initial_radiation = init_rad
            self.initial_efficiency = init_eff
            print(f"【基准】寿命: {init_life:.2e}s | 辐射: {init_rad:.2f}W | 效率: {init_eff*100:.2f}%")
        except Exception as e:
            print(f"基准提取失败: {e}")
            self.initial_life, self.initial_radiation, self.initial_efficiency = 100.0, 50.0, 0.5

        # 缓存初始物理场
        self.base_temp_map = None
        self.base_lifespan_map = None
        self._cache_baseline_fields()

    def _cache_baseline_fields(self):
        print("正在缓存初始圆柱体的 3D 物理场 (仅执行一次)...")
        # 临时生成一个标准圆柱体
        binary_grid = initialize_cylinder_grid()
        self.density_field = binary_grid.astype(np.float32)
        self.current_voxel_grid = binary_grid.copy()
        
        try:
            temp_map, _, _, max_temp = self._run_comsol_simulation_3d()
            evap_map = calculate_evaporation_rate(temp_map)
            self.base_temp_map = temp_map
            self.base_lifespan_map = self._calculate_lifespan_map(self.current_voxel_grid, evap_map)
            print(f"基准场缓存成功！中心最高温度约: {max_temp:.2f}K")
        except Exception as e:
            print(f"缓存基准场失败: {e}，使用假数据")
            # 假数据
            self.base_temp_map = np.ones(self.grid_size, dtype=np.float32) * 2500.0
            self.base_lifespan_map = np.ones(self.grid_size, dtype=np.float32) * 1e5

    def _initialize_masks(self):
        nx, ny, nz = self.grid_size
        self.frozen_mask = np.zeros((nx, ny, nz), dtype=bool)
        radius = 25 
        x, y, z = np.ogrid[:nx, :ny, :nz]
        dist_sq = (x - nx//2)**2 + (y - ny//2)**2
        # 固定上下电极
        self.frozen_mask[(dist_sq <= radius**2) & (z < 2)] = True
        self.frozen_mask[(dist_sq <= radius**2) & (z > nz - 3)] = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        # 恢复几何与密度到初始圆柱体状态
        binary_grid = initialize_cylinder_grid()
        self.valid_mask = ~self.frozen_mask
        self.target_volume = np.sum(binary_grid[self.valid_mask])
        self.density_field = binary_grid.astype(np.float32)
        self.current_voxel_grid = binary_grid.copy()
        self.step_count = 0
        
        # 复用初始物理场
        return self._get_obs(temp=self.base_temp_map, lifespan_map=self.base_lifespan_map), {}

    def _apply_density_action(self, action):
        action_gpu = cp.asarray(action)
        zoom_factors = (self.grid_size[0]/self.action_shape[0], 
                self.grid_size[1]/self.action_shape[1], 
                self.grid_size[2]/self.action_shape[2])
        delta_rho = c_zoom(action_gpu, zoom_factors, order=1) 
        delta_rho = c_gaussian(delta_rho, sigma=1.2)
        delta_rho = cp.asnumpy(delta_rho) 

        # 释放内存
        del action_gpu
        cp.get_default_memory_pool().free_all_blocks()
    
        # 防止形状不匹配
        target_shape = self.grid_size
        pad_width = []
        for i in range(3):
          diff = target_shape[i] - delta_rho.shape[i]
          if diff > 0:
            pad_width.append((0, diff))
          elif diff < 0:
            pad_width.append((0, 0))
            delta_rho = delta_rho[(slice(None),) * i + (slice(0, target_shape[i]),)]
          else:
            pad_width.append((0, 0))
        if any(p[1] > 0 for p in pad_width):
          delta_rho = np.pad(delta_rho, pad_width, mode='edge')
    
        raw_density = self.density_field + delta_rho
        
        # 二分法体积守恒投影
        def calc_vol(offset):
            projected = np.clip(raw_density[self.valid_mask] + offset, 0.0, 1.0)
            return np.sum(projected)

        l_min, l_max = -2.0, 2.0
        for _ in range(40): 
            l_mid = (l_min + l_max) / 2.0
            if calc_vol(l_mid) > self.target_volume: l_max = l_mid
            else: l_min = l_mid
                
        optimal_offset = (l_min + l_max) / 2.0
        self.density_field[self.valid_mask] = np.clip(raw_density[self.valid_mask] + optimal_offset, 0.0, 1.0)
        
        # 保护电极
        self.density_field[self.frozen_mask & (self.current_voxel_grid == 1)] = 1.0
        self.density_field[self.frozen_mask & (self.current_voxel_grid == 0)] = 0.0

        self.current_voxel_grid = (self.density_field >= 0.5).astype(np.float32)

    def _extract_mesh(self):
        verts, faces, _, _ = measure.marching_cubes(self.density_field, level=0.5)
        mesh = trimesh.Trimesh(vertices=verts * self.voxel_size, faces=faces)
        trimesh.smoothing.filter_laplacian(mesh, iterations=2)
        return mesh

    def _run_comsol_simulation_3d(self):
        if self.model is None:
            print("未找到基准模型，用假数据调试代码")
            return np.ones(self.grid_size)*2500.0, 50.0, 100.0, 2500.0

        model = self.model
        try:
            mesh = self._extract_mesh()
            stl_path = os.path.join(self.output_dir, 'temp_topo.stl')
            mesh.export(stl_path)

            geom = model.java.component("comp1").geom("geom1")
            geom.feature("imp1").set("filename", stl_path)
            geom.run()
            model.java.component("comp1").mesh(str(model.java.component("comp1").mesh().tags()[0])).run()
            model.java.study(str(model.java.study().tags()[0])).run()

            net_rad = float(model.evaluate('Total_Rad'))       
            p_in = float(model.evaluate('P_in'))           
            max_temp = float(model.evaluate('Max_Temp')) 
            
            # 提取温度场
            x_c, y_c, z_c = model.evaluate('x')*1000, model.evaluate('y')*1000, model.evaluate('z')*1000
            t_p = model.evaluate('T')
            x_g, y_g, z_g = np.meshgrid(np.arange(self.grid_size[0]), np.arange(self.grid_size[1]), np.arange(self.grid_size[2]), indexing='ij')
            
            temp_map = griddata((x_c.ravel(), y_c.ravel(), z_c.ravel()), t_p.ravel(), 
                                (x_g*self.voxel_size, y_g*self.voxel_size, z_g*self.voxel_size), method='nearest')
            temp_map = np.nan_to_num(temp_map.astype(np.float32), nan=300.0)
            # 清理COMSOL缓存
            model.java.mesh(str(model.java.component("comp1").mesh().tags()[0])).clear()
            model.java.sol("sol1").clearSolutionData()
            return temp_map, net_rad, p_in, max_temp
            
        except Exception as e:
            raise RuntimeError(f"COMSOL Error: {str(e)[:80]}")

    def _check_structural_integrity(self):
        labeled_array, num_features = ndimage.label(self.current_voxel_grid)
        return num_features == 1

    def step(self, action):
        self.step_count += 1
        self._apply_density_action(action)
            
        if not self._check_structural_integrity():
            return self._get_obs(), -50.0, True, False, {"error": "Fragmentation"}
            
        try:
            temp_map, net_rad, p_in, max_temp = self._run_comsol_simulation_3d()
        except Exception as e:
            return self._get_obs(), -80.0, True, False, {"error": "COMSOL Fail"}
            
        if max_temp > 3273.15:
            return self._get_obs(temp_map=temp_map), -50.0, True, False, {"error": "Temp > 3000C"}
            
        evap_map = calculate_evaporation_rate(temp_map) 
        
        # 计算全空间寿命图谱
        lifespan_map = self._calculate_lifespan_map(self.current_voxel_grid, evap_map)
        topo_life = np.min(lifespan_map[self.current_voxel_grid > 0.5]) # 提取最短寿命用于奖励
        
        efficiency = net_rad / p_in if p_in > 1e-5 else 0.0 

        if self.step_count % 10 == 0: self._save_evolution_frame(efficiency, topo_life)
        reward = self._calculate_reward(efficiency, topo_life)
        
        # 传入 temp_map 和 lifespan_map
        obs = self._get_obs(temp_map, lifespan_map)
        
        truncated = bool(self.step_count >= self.max_steps)
        return obs, reward, False, truncated, {"life": topo_life, "rad": net_rad, "eff": efficiency}
        
    def _calculate_reward(self, efficiency, topo_life):
        # 计算奖励
        # 能量转换效率增益 (目标提升>30%)
        eff_gain = (efficiency - self.initial_efficiency) / self.initial_efficiency
        r_eff = 15.0 * np.tanh(4.0 * eff_gain) 
        
        # 寿命约束与激励 (目标 > 50%)
        life_ratio = topo_life / self.initial_life
        if life_ratio < 0.3:
            # 触碰红线给予平方级惩罚
            r_life = -100.0 * ((0.3 - life_ratio) / 0.3) ** 2
        else:
            # 鼓励向50%以上突破
            r_life = 10.0 * np.tanh(3.0 * (life_ratio - 0.3))
            
        return float(r_eff + r_life)

    def _save_evolution_frame(self, efficiency, topo_life):
        verts, faces, _, _ = measure.marching_cubes(self.density_field, level=0.5)
        mesh = trimesh.Trimesh(vertices=verts * self.voxel_size, faces=faces)
        mesh.export(os.path.join(self.output_dir, f"frame_{self.step_count:05d}.stl"))
        with open(os.path.join(self.output_dir, "log.csv"), 'a') as f:
            f.write(f"{self.step_count},{efficiency:.4f},{topo_life:.2e}\n")

    def _get_obs(self, temp=None, lifespan_map=None):
        target_shape = (50, 50, 75)        
        zoom_factor = (
            target_shape[0] / self.grid_size[0],
            target_shape[1] / self.grid_size[1],
            target_shape[2] / self.grid_size[2]
            )
        
        #  Ch 0: SDF
        voxel_gpu = cp.asarray(self.current_voxel_grid)
        inside_dist = c_edt(voxel_gpu)
        outside_dist = c_edt(1 - voxel_gpu)
        sdf_full = cp.asnumpy(inside_dist - outside_dist)
        
        # 释放GPU
        del voxel_gpu, inside_dist, outside_dist
        cp.get_default_memory_pool().free_all_blocks()

        # 降采样 SDF
        sdf_down = ndimage.zoom(sdf_full, zoom_factor, order=1)
        
        # SDF 归一化
        max_dist = self.grid_size[0] / 2.0
        sdf_norm = np.clip(sdf_down / max_dist, -1.0, 1.0) 
        
        #  Ch 1: 温度
        temp_norm = np.zeros(target_shape, dtype=np.float32)
        if temp is not None: 
            # 降采样温度场
            temp_down = ndimage.zoom(temp, zoom_factor, order=1)
            temp_norm = np.clip(temp_down / 3500.0, 0.0, 1.0)
        
        # Ch 2: 寿命
        lifespan_norm = np.zeros(target_shape, dtype=np.float32)
        if lifespan_map is not None:
            # 降采样寿命场
            lifespan_down = ndimage.zoom(lifespan_map, zoom_factor, order=1)
            safe_lifespan = np.maximum(lifespan_down, 1.0) 
            lifespan_norm = np.clip((np.log10(safe_lifespan) - 1.0) / 4.0, 0.0, 1.0)
            
        obs = np.stack([sdf_norm, temp_norm, lifespan_norm], axis=0).astype(np.float32)
        
        return obs

    def _calculate_lifespan_map(self, binary_grid, evap_map):
        binary_gpu = cp.asarray(binary_grid)
        edt_dist = cp.asnumpy(c_edt(binary_gpu))
        del binary_gpu
        cp.get_default_memory_pool().free_all_blocks()
    
        valid_mask = (binary_grid > 0.5)
        safe_evap = np.maximum(evap_map, 1e-15)
        lifespan_map = np.zeros_like(evap_map)
        lifespan_map[valid_mask] = (0.1 * edt_dist[valid_mask] * (VOXEL_SIZE / 1000.0) * RHO_DENSITY) / safe_evap[valid_mask]
        lifespan_map[~valid_mask] = 0.0
        return lifespan_map

class Topo_3DCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(Topo_3DCNN, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Flatten()
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

    # 主程序
if __name__ == "__main__":
    torch.set_num_threads(16) 
    print("正在启动 COMSOL 客户端...")
    global_client = mph.start(cores=16)

    # 初始化环境
    env = TungstenTopologyEnv(global_client)
    init_life = env.initial_life 
    init_rad = env.initial_radiation 
    init_eff = env.initial_efficiency
    print(f"【基准】寿命: {init_life:.2e}s | 辐射: {init_rad:.2f}W | 效率: {init_eff*100:.2f}%")

    policy_kwargs = dict(
        features_extractor_class=Topo_3DCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False 
    )
    
    model = PPO(
        "CnnPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        n_steps=128,
        batch_size=16, 
        learning_rate=3e-4,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1, 
        device="cuda",
        tensorboard_log="./tungsten_ppo_tensorboard/"
    )
    
    print("开始强化学习训练...")
    model.learn(total_timesteps=100)
    model.save("ppo_tungsten_3D__optimized")
    
    # 训练结束
    global_client.clear()
    print("训练完成，模型与演化日志已保存。")