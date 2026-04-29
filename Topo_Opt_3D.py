import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mph
import os
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
    extract_mesh,calculate_lifespan_map
)
from logger_utils import setup_logger


# 初始化日志记录器
logger = setup_logger(__name__)

class TungstenTopologyEnv(gym.Env):
    def __init__(self, client):
        super(TungstenTopologyEnv, self).__init__()
        self.client = client
        self.grid_size = GRID_SIZE
        self.voxel_size = VOXEL_SIZE
        self.rho_density = RHO_DENSITY
        self.target_shape = (50, 50, 75)
        self.max_steps = 1000 
        current_path = os.path.dirname(os.path.abspath(__file__))
            
        # 状态空间为3通道: [SDF边界, 温度场, 寿命场]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(3, self.target_shape[0], self.target_shape[1], self.target_shape[2]), 
            dtype=np.float32
        )
        # 3D 动作空间需 6 个参数：(x1, y1, z1) 用于挖除，(x2, y2, z2) 用于填补
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        self.step_count = 0
        self.output_dir = "./topology_evolution"
        os.makedirs(self.output_dir, exist_ok=True)
        self.current_voxel_grid = self._initialize_masks()

        # 初始化指标
        self.initial_life = 7.48e5
        self.initial_radiation = 423.34
        self.initial_efficiency = 1.0
        self.base_temp_map = None
        self.base_lifespan_map = None

        # 基准评估
        self._initialize_baseline()

    def _initialize_baseline(self):
        logger.info("正在提取基准性能...")
        current_path = os.path.dirname(os.path.abspath(__file__))
        init_model_path = os.path.join(current_path, 'Topo_Init_3D.mph')

        if not os.path.exists(init_model_path):
            raise FileNotFoundError(f"找不到基准模型: {init_model_path}")

        # 加载基准模型
        init_model = self.client.load(init_model_path)
        
        try:
            mesh_tags = init_model.java.component("comp1").mesh().tags()
            if len(mesh_tags) > 0:
                init_model.java.component("comp1").mesh(mesh_tags[0]).run()
                
            study_tags = list(init_model.java.study().tags())
            if len(study_tags) > 0:
                init_model.java.study(study_tags[0]).run()
            
            # 提取性能
            net_rad = float(init_model.evaluate('Total_Rad'))
            p_in = float(init_model.evaluate('P_in'))
            
            # 提取物理场
            x_coords = init_model.evaluate('x')
            y_coords = init_model.evaluate('y')
            z_coords = init_model.evaluate('z')
            temp_points = init_model.evaluate('T')
            
            nx, ny, nz = self.grid_size
            x_g, y_g, z_g = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
            
            # 插值到网格上
            temp_map = griddata(
                (x_coords.ravel(), y_coords.ravel(), z_coords.ravel()), 
                temp_points.ravel(), 
                (x_g * self.voxel_size, y_g * self.voxel_size, z_g * self.voxel_size), 
                method='linear'
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
        self.base_lifespan_map = calculate_lifespan_map(binary_grid, evap_map)
        self.initial_life = np.min(self.base_lifespan_map[binary_grid > 0.5])
        
        logger.info(
            f"基准参数 | "
            f"寿命：{self.initial_life:.2e}s | "
            f"辐射：{self.initial_radiation:.2f}W | "
            f"效率：{self.initial_efficiency*100:.2f}%"
        )

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

    def _apply_action(self, action, brush_radius):
        # 替换以两个坐标为中心的球形体素区域

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
                        cx1, cy1, cz1 = x1 + dx, y1 + dy, z1 + dz
                        if 0 <= cx1 < nx and 0 <= cy1 < ny and 0 <= cz1 < nz:
                            if self.current_voxel_grid[cx1, cy1, cz1] == 1.0 and not self.frozen_mask[cx1, cy1, cz1]:
                                rm_candidates.append((cx1, cy1, cz1))
                        cx2, cy2, cz2 = x2 + dx, y2 + dy, z2 + dz
                        if 0 <= cx2 < nx and 0 <= cy2 < ny and 0 <= cz2 < nz:
                            if self.current_voxel_grid[cx2, cy2, cz2] == 0.0 and not self.frozen_mask[cx2, cy2, cz2]:
                                add_candidates.append((cx2, cy2, cz2))

        swap_count = min(len(rm_candidates), len(add_candidates))
        if swap_count == 0: return False, 0

        for i in range(swap_count):
            self.current_voxel_grid[rm_candidates[i]] = 0.0
            self.current_voxel_grid[add_candidates[i]] = 1.0

        self.current_voxel_grid = ndimage.binary_fill_holes(self.current_voxel_grid)
        self.current_voxel_grid = ndimage.grey_erosion(self.current_voxel_grid, size=1)

        return True, swap_count

    def _run_comsol_simulation_3d(self):
    # 运行仿真
        # 加载优化模型
        current_path = os.path.dirname(os.path.abspath(__file__))
        opt_model_path = os.path.join(current_path, 'Topo_Opt_3D.mph')
        self.opt_model = self.client.load(opt_model_path) if os.path.exists(opt_model_path) else None

        nx, ny, nz = self.grid_size
        
        if self.opt_model is None:
             raise RuntimeError("未找到优化模型...")

        try:
            mesh = extract_mesh(self)
            stl_path = os.path.join(self.output_dir, 'temp_topo.stl')
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
                method='linear'
            ).astype(np.float32)
            
            temp_map = np.nan_to_num(temp_map, nan=300.0)
            
            return temp_map, float(net_rad), float(p_in), float(max_temp)
            
        except Exception as e: 
            pass
            raise RuntimeError(f"COMSOL Error: {str(e)}")

    def _check_structural_integrity(self):
        structure = ndimage.generate_binary_structure(3, 3) 
        labeled_array, num_features = ndimage.label(self.current_voxel_grid, structure=structure)
        if num_features > 1:
            return False
        
        return True

    def step(self, action):
        self.step_count += 1

        # 修改拓扑
        success, modified_count = self._apply_action(action, brush_radius=3)
        if not success:
            return self._get_obs(), -2.0, False, False, {"error": "Invalid action"}
            
        # 连通性检查,若断裂，返回错误信息，并给惩罚
        if not self._check_structural_integrity():
            obs = self._get_obs(temp=None, lifespan_map=None) 
            return obs, -20.0, True, False, {"error": "Fragmentation"}
            
        # 执行COMSOL仿真
        try:
            temp_map, net_rad, p_in, max_temp = self._run_comsol_simulation_3d()
        except Exception as e:
            logger.error(f"COMSOL 运行失败！错误信息: {e}")
            obs = self._get_obs(temp=None, lifespan_map=None)
            return obs, -20.0, True, False, {"error": "COMSOL Fail"}
            
        if max_temp > 3273.15:
            logger.warning(f"温度超过限制: {max_temp - 273.15:.1f}℃")
            return self._get_obs(temp=temp_map), -10.0, True, False, {"error": "Temp > 3000C"}

        logger.info(f"最高温度: {max_temp - 273.15:.1f}℃, 辐射功率: {net_rad:.2f}W")
            
        evap_map = calculate_evaporation_rate(temp_map) 
        lifespan_map = calculate_lifespan_map(self.current_voxel_grid, evap_map)
        topo_life = np.min(lifespan_map[self.current_voxel_grid > 0.5])
        
        efficiency = net_rad / p_in if p_in > 1e-5 else 0.0 

        if self.step_count % 10 == 0: self._save_evolution_frame(net_rad, efficiency, topo_life)
        reward = self._calculate_reward(net_rad, efficiency, topo_life)
        
        obs = self._get_obs(temp=temp_map, lifespan_map=lifespan_map)
        
        truncated = bool(self.step_count >= self.max_steps)
        return obs, reward, False, truncated, {"life": topo_life, "rad": net_rad, "eff": efficiency}

    def _calculate_reward(self, net_rad, efficiency, topo_life):
        # 1辐射功率奖励
        rad_gain = (net_rad - self.initial_radiation) / self.initial_radiation
        r_rad = np.clip(20.0 * rad_gain, -10.0, 10.0)
    
        # 2效率提升奖励
        eff_gain = (efficiency - self.initial_efficiency) / self.initial_efficiency
        r_eff = np.clip(10.0 * eff_gain, -5.0, 5.0)
    
        # 3器件寿命奖励
        life_ratio = topo_life / self.initial_life
        if life_ratio < 0.3:
            r_life =np.clip((-10.0 * ((0.3 - life_ratio)/ 0.3)),-10.0,0.0)
        else:
            r_life = np.clip((3.0 * ((life_ratio - 0.3)/0.3)), 0.0, 3.0)
        
        return float(r_rad + r_eff + r_life)

    def _save_evolution_frame(self, net_rad, efficiency, topo_life):
        logger.debug(f"保存演化帧 step={self.step_count}")

        # 格式化性能指标
        # 辐射功率
        rad_str = f"{net_rad:.2f}"

        # 效率
        eff_percent = efficiency * 100.0
        eff_str = f"{eff_percent:.2f}"

        # 寿命:x分y秒
        total_seconds = int(topo_life)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        life_str = f"{minutes}m{seconds:05.2f}s"

        # 保存STL(包含指标摘要)
        smoothed_grid = ndimage.gaussian_filter(self.current_voxel_grid, sigma=1.0)
        verts, faces, normals, values = measure.marching_cubes(smoothed_grid, 0.5)
        mesh = extract_mesh(self)
        stl_filename = f"frame_{self.step_count:05d}_rad{float(rad_str):.0f}W_life{minutes}m.stl"
        mesh.export(os.path.join(self.output_dir, stl_filename))

        # 记录性能指标
        log_path = os.path.join(self.output_dir, "performance_log.csv")
        file_exists = os.path.exists(log_path)
        with open(log_path, 'a') as f:
            if not file_exists:
                f.write("step,net_rad(W),efficiency(%),topo_life(mm:ss)\n")
            f.write(f"{self.step_count},{rad_str},{eff_str},{life_str}\n")

        logger.info(
            f"已保存演化拓扑 | Step: {self.step_count} | "
            f"辐射：{rad_str}W | 效率：{eff_str}% | 寿命：{life_str}"
        )

    def _get_obs(self, temp=None, lifespan_map=None, target_shape = None):    
        if target_shape is None:
            target_shape = self.target_shape

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
        del voxel_gpu, inside_dist, outside_dist

        
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
    torch.set_num_threads(2) 
    logger.info("正在启动 COMSOL 客户端...")
    global_client = mph.start(cores=12)

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
    
    # 自动检测 CUDA 可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备：{device}")

    model = PPO(
        "CnnPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        n_steps=256,
        batch_size=64, 
        learning_rate=1.5e-4,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef= 0.01 ,
        verbose=1, 
        device=device,
        tensorboard_log="./tungsten_ppo_tensorboard/"
    )
    
    logger.info("开始强化学习训练...")
    model.learn(total_timesteps=1000)
    model.save("ppo_tungsten_3D__optimized")
    
    # 训练结束
    global_client.clear()
    logger.info("训练完成，模型与演化日志已保存。")