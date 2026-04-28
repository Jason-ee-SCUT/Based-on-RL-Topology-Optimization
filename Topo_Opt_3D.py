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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from physics_tools import (
    GRID_SIZE, VOXEL_SIZE, RHO_DENSITY,
    initialize_cylinder_grid, calculate_evaporation_rate, 
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
            
        # 状态空间为3通道: [SDF边界, 温度场, 寿命场]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(3, 50, 50, 75),
            dtype=np.float32
            )
        
        # 动作空间，每个子动作对应一个粗网格的密度调整，最终通过插值和平滑转换为细网格的密度变化
        self.num_sub_actions = 5
        self.sub_action_shape = (self.grid_size[0]//10, self.grid_size[1]//10, self.grid_size[2]//10)
        self.action_space = spaces.Box(
            low=-0.15, high=0.15, 
            shape=(self.num_sub_actions, *self.sub_action_shape), 
            dtype=np.float32
        )
        
        self.step_count = 0
        self.output_dir = "./topology_evolution"
        os.makedirs(self.output_dir, exist_ok=True)
        self.target_volume = 0.0
        self._initialize_masks()

        # 加载优化模型
        current_path = os.path.dirname(os.path.abspath(__file__))
        opt_model_path = os.path.join(current_path, 'Topo_Opt_3D.mph')
        self.opt_model = self.client.load(opt_model_path) if os.path.exists(opt_model_path) else None

        # 初始化指标
        self.initial_life = 7.48e5
        self.initial_radiation = 423.34
        self.initial_efficiency = 1.0
        self.base_temp_map = None
        self.base_lifespan_map = None

        # 基准评估
        self._initialize_baseline()

    def _initialize_baseline(self):
        print("正在提取基准性能...")
        current_path = os.path.dirname(os.path.abspath(__file__))
        init_model_path = os.path.join(current_path, 'Topo_Init_3D.mph')

        if not os.path.exists(init_model_path):
            raise FileNotFoundError(f"找不到基准模型: {init_model_path}")

        # 加载基准模型
        init_model = self.client.load(init_model_path)
        
        try:
            mesh_tags = init_model.java.component("comp1").mesh().tags()
            if len(mesh_tags) > 0:
                init_model.java.component("comp1").mesh(str(mesh_tags[0])).run()
                
            study_tags = init_model.java.study().tags()
            init_model.java.study(str(study_tags[0])).run()
            
            # 提取性能
            net_rad = float(init_model.evaluate('Total_Rad'))
            p_in = float(init_model.evaluate('P_in'))
            
            # 提取物理场
            x_coords = init_model.evaluate('x') * 1000
            y_coords = init_model.evaluate('y') * 1000
            z_coords = init_model.evaluate('z') * 1000
            temp_points = init_model.evaluate('T')
            
            nx, ny, nz = self.grid_size
            x_g, y_g, z_g = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
            
            # 插值到网格上
            temp_map = griddata(
                (x_coords.ravel(), y_coords.ravel(), z_coords.ravel()), 
                temp_points.ravel(), 
                (x_g * self.voxel_size, y_g * self.voxel_size, z_g * self.voxel_size), 
                method='nearest'
            ).astype(np.float32)
            temp_map = np.nan_to_num(temp_map, nan=300.0)
            
        finally:
            # 释放内存
            init_model.clear()
            del init_model

        # 计算并保存基准指标
        self.initial_radiation = net_rad
        self.initial_efficiency = (net_rad / p_in) if p_in > 1e-5 else 0.0
        
        # 5. 计算并缓存三维图谱
        evap_map = calculate_evaporation_rate(temp_map)
        binary_grid = initialize_cylinder_grid()
        
        self.base_temp_map = temp_map
        self.base_lifespan_map = self._calculate_lifespan_map(binary_grid, evap_map)
        self.initial_life = np.min(self.base_lifespan_map[binary_grid > 0.5])
        
        print(f"【基准参数】寿命: {self.initial_life:.2e}s | 辐射: {self.initial_radiation:.2f}W | 效率: {self.initial_efficiency*100:.2f}%")

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
        
        # 复用基准指标
        return self._get_obs(temp=self.base_temp_map, lifespan_map=self.base_lifespan_map), {}

    def _apply_density_action(self, total_delta_rho):
        # total_delta_rho是(100, 100, 150) 张量，代表5个子动作叠加后的总密度变化意图。
        raw_density = self.density_field + total_delta_rho
        
        # 二分法投影
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
        # 提取等值面
        verts, faces, _, _ = measure.marching_cubes(self.density_field, level=0.5)
        mesh = trimesh.Trimesh(vertices=verts * self.voxel_size, faces=faces)

        # 修复网格拓扑
        mesh.remove_duplicate_faces()      # 删除重复面
        mesh.remove_degenerate_faces()     # 删除面积为0的退化面
        mesh.remove_unreferenced_vertices()# 删除孤立的顶点
        mesh.fill_holes()                  # 封闭微小的孔洞
        mesh.fix_normals()                 # 法线向外
        trimesh.smoothing.filter_laplacian(mesh, iterations=2)# 平滑体素

        return mesh

    def _run_comsol_simulation_3d(self):
        
        if self.opt_model is None:
             raise RuntimeError("未找到优化模型...")

        model = self.opt_model
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
            model.java.sol("sol1").clearSolutionData()
            return temp_map, net_rad, p_in, max_temp
            
        except Exception as e:
            raise RuntimeError(f"COMSOL Error: {str(e)[:80]}")

    def _check_structural_integrity(self):
        labeled_array, num_features = ndimage.label(self.current_voxel_grid)
        return num_features == 1

    def step(self, macro_action):
        self.step_count += 1
        
        # 初始化总增量场
        total_delta_rho = np.zeros(self.grid_size, dtype=np.float32)
        
        # 遍历5个子动作，分别上采样并累加
        zoom_factors = (self.grid_size[0]/self.sub_action_shape[0], 
                        self.grid_size[1]/self.sub_action_shape[1], 
                        self.grid_size[2]/self.sub_action_shape[2])
        
        for i in range(self.num_sub_actions):
            sub_action = macro_action[i] # shape: (10, 10, 15)
            
            delta_rho = ndimage.zoom(sub_action, zoom_factors, order=1) 
            delta_rho = ndimage.gaussian_filter(delta_rho, sigma=1.5)
            
            # 处理因为缩放导致的边缘像素差
            delta_rho = delta_rho[:self.grid_size[0], :self.grid_size[1], :self.grid_size[2]]
            
            # 累加到总增量中
            total_delta_rho += delta_rho
            
        # 应用总增量，并约束体积
        self._apply_density_action(total_delta_rho)
            
        # 连通性检查
        if not self._check_structural_integrity():
            # 如果断裂，使用上一次成功的物理场返回错误信息，并给惩罚
            obs = self._get_obs(temp=None, lifespan_map=None) 
            return obs, -10.0, True, False, {"error": "Fragmentation"}
            
        # 执行COMSOL仿真
        try:
            temp_map, net_rad, p_in, max_temp = self._run_comsol_simulation_3d()
        except Exception as e:
            print(f"COMSOL 运行失败！错误信息: {e}")
            obs = self._get_obs(temp=None, lifespan_map=None)
            return obs, -20.0, True, False, {"error": "COMSOL Fail"}
            
        if max_temp > 3273.15:
            return self._get_obs(temp=temp_map), -15.0, True, False, {"error": "Temp > 3000C"}

        print(f"最高温度: {max_temp - 273.15:.1f}℃, 辐射功率: {net_rad:.2f}W")
            
        evap_map = calculate_evaporation_rate(temp_map) 
        lifespan_map = self._calculate_lifespan_map(self.current_voxel_grid, evap_map)
        topo_life = np.min(lifespan_map[self.current_voxel_grid > 0.5])
        
        efficiency = net_rad / p_in if p_in > 1e-5 else 0.0 

        if self.step_count % 10 == 0: self._save_evolution_frame(efficiency, topo_life)
        reward = self._calculate_reward(efficiency, topo_life)
        
        obs = self._get_obs(temp=temp_map, lifespan_map=lifespan_map)
        
        truncated = bool(self.step_count >= self.max_steps)
        return obs, reward, False, truncated, {"life": topo_life, "rad": net_rad, "eff": efficiency}

    def _calculate_reward(self, net_rad, topo_life, efficiency):
        # 1辐射功率奖励
        rad_gain = (net_rad - self.initial_radiation) / self.initial_radiation
        r_rad = np.clip(20.0 * rad_gain, -10.0, 10.0)
    
        # 2效率提升奖励
        eff_gain = (efficiency - self.initial_efficiency) / self.initial_efficiency
        r_eff = np.clip(10.0 * eff_gain, -5.0, 5.0)
    
        # 3器件寿命奖励
        life_ratio = topo_life / self.initial_life
        if life_ratio < 0.3:
            r_life =np.clip(-10.0 * ((0.3 - life_ratio)/ 0.3)),-10.0,0.0)
        else:
            r_life = np.clip((3.0 * ((life_ratio - 0.3)/0.3)), 0.0, 3.0)
        
        return float(r_rad + r_eff + r_life)

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
        inside_dist = ndimage.distance_transform_edt(self.current_voxel_grid)
        outside_dist = ndimage.distance_transform_edt(1 - self.current_voxel_grid)
        sdf_full = inside_dist - outside_dist
        
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
        edt_dist = ndimage.distance_transform_edt(binary_grid)
    
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

    policy_kwargs = dict(
        features_extractor_class=Topo_3DCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False 
    )
    
    model = PPO(
        "CnnPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        n_steps=256,
        batch_size=32, 
        learning_rate=1.5e-4,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef= "auto" ,
        verbose=1, 
        device="cuda",
        tensorboard_log="./tungsten_ppo_tensorboard/"
    )
    
    print("开始强化学习训练...")
    model.learn(total_timesteps=10000)
    model.save("ppo_tungsten_3D__optimized")
    
    # 训练结束
    global_client.clear()
    print("训练完成，模型与演化日志已保存。")