import numpy as np;
import gymnasium as gym;
from gymnasium import spaces;
import mph;
import os;
import scipy.ndimage as ndimage;
from scipy.interpolate import griddata;
from scipy.ndimage import distance_transform_edt;
from skimage.morphology import skeletonize;
from skimage import measure  ;
from stable_baselines3 import PPO;
from sklearn.cluster import DBSCAN;
import jpype;

class TungstenTopologyEnv(gym.Env):
    def __init__(self, grid_size=(50, 200)):
        super(TungstenTopologyEnv, self).__init__()
        self.grid_size = grid_size
        
        # 1 初始化 COMSOL 通信
        print("正在启动 COMSOL 后台服务器...")
        self.client = mph.start(cores=8)
        
        # 加载基础模型文件
        current_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_path,'Topo_opt_2D.mph')
        if os.path.exists(model_path):
            self.model = self.client.load(model_path)
        else:
            print(f"【警告】未找到 {model_path}，使用随机假数据模式供调试代码使用。")
            self.model = None
            
        self._dynamic_geom_tags = []
        self.voxel_size = 0.1e-3 # 每个体素代表 0.1mm 的实际尺寸
        
        # 2 状态空间与动作空间
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(3, grid_size[0], grid_size[1]), 
            dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # 3 初始化参数
        self.initial_radiation = 1.0 
        self.initial_life = 100.0
        self.rho_density = 19350.0
        self.current_voxel_grid = self._initialize_cylinder()
        
        # 启动时残余几何扫荡
        self._cleanup_comsol_geometry()

    # 清理历史几何
    def _cleanup_comsol_geometry(self):
        if self.model is None:
            return
            
        geom = self.model.java.component("comp1").geom("geom1")
        
        # 1 清理本回合特征
        for tag in self._dynamic_geom_tags:
            try:
                geom.feature().remove(tag)
            except Exception:
                pass
        self._dynamic_geom_tags = []
        
        # 2 清理残留节点
        for i in range(500):
            try:
                geom.feature().remove(f"rl_poly_{i}")
            except Exception:
                pass
                
        # 刷新内部结构
        geom.run()

    def _initialize_cylinder(self):
        nr, nz = self.grid_size
        grid = np.zeros((nr, nz), dtype=np.float32)
        radius = 25  
        z_start = (nz - 150) // 2
        z_end = z_start + 150
        
        r, z = np.ogrid[:nr, :nz]
        cylinder_mask = (r <= radius) & (z >= z_start) & (z <= z_end)       
        grid[cylinder_mask] = 1.0       
        
        self.frozen_mask = np.zeros_like(grid, dtype=bool)
        electrode_thickness = 1 
        bottom_electrode = (r <= radius) & (z >= z_start) & (z < z_start + electrode_thickness)
        top_electrode = (r <= radius) & (z > z_end - electrode_thickness) & (z <= z_end)
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
        dbscan = DBSCAN(eps=1.5, min_samples=2)
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
        A = 3.9e6     
        B = -1.023e5  
        evap_map = A * np.exp(B / safe_temp) 
        return evap_map

    # 进行COMSOL仿真，返回温度分布和净辐射增益
    def _run_comsol_simulation_2d(self, grid_2d):
        nr, nz = self.grid_size
        
        if self.model is None:
            return np.ones((nr, nz), dtype=np.float32) * 2500.0, 50.0

        smoothed_grid = ndimage.gaussian_filter(grid_2d, sigma=1.0)
        contours = measure.find_contours(smoothed_grid, 0.5)
        
        try:
            # 几何组件定义
            geom = self.model.java.component("comp1").geom("geom1")
            
            # 清理旧几何
            self._cleanup_comsol_geometry()
            
            for i, contour in enumerate(contours):
                # 过滤噪点，防止少于3个点无法形成多边形导致崩溃
                if len(contour) < 3:
                    continue
                    
                tag = f"rl_poly_{i}"
                self._dynamic_geom_tags.append(tag)
                
                geom.feature().create(tag, "Polygon")
                
                r_coords = contour[:, 0] * self.voxel_size
                z_coords = contour[:, 1] * self.voxel_size
                
                # jpype 强转 Java 数组，解决数据类型拒收报错
                r_java = jpype.JArray(jpype.JDouble)(r_coords.tolist())
                z_java = jpype.JArray(jpype.JDouble)(z_coords.tolist())
                
                geom.feature(tag).set("x", r_java)
                geom.feature(tag).set("y", z_java)
                geom.feature(tag).set("type", "solid") 

            # 更新几何
            geom.run()

            # 网格操作
            try:
                mesh_tags = self.model.java.component("comp1").mesh().tags()
                
                # 检查是否存在网格
                if len(mesh_tags) == 0:
                    self.model.java.component("comp1").mesh().create("mesh1")
                    target_mesh = "mesh1"
                else:
                    # 获取第一个网格的标签并强转为 Python 字符串
                    target_mesh = str(mesh_tags[0])
                
                # Java API执行网格剖分
                self.model.java.component("comp1").mesh(target_mesh).run()
                
            except Exception as e:
                print(f"网格剖分失败，可能是出现了极度狭窄的锐角: {e}")
                raise e
            
            # 求解操作
            try:
                # 获取所有研究标签
                study_tags = self.model.java.study().tags()
                
                if len(study_tags) == 0:
                    raise ValueError("模型中丢失了研究(Study)节点，请检查基准模型！")
                    
                target_study = str(study_tags[0])
                
                # 执行求解
                self.model.java.study(target_study).run()
                
            except Exception as e:
                print(f"求解器运行失败: {e}")
                raise e

            # 提取物理场数据
            net_rad = self.model.evaluate('Total_Rad')
            r_coords_comsol = self.model.evaluate('r')
            z_coords_comsol = self.model.evaluate('z')
            temp_points = self.model.evaluate('T')
            # 提取 COMSOL 内部寻优得到的最佳电压
            optimal_voltage = self.model.evaluate('V_in') 
            print(f"当前拓扑最佳工作电压: {optimal_voltage:.2f} V")

            # 网格化插值
            r_grid, z_grid = np.meshgrid(np.arange(nr), np.arange(nz), indexing='ij')
            r_grid_m = r_grid * self.voxel_size
            z_grid_m = z_grid * self.voxel_size
            
            temp_map = griddata(
                (r_coords_comsol.ravel(), z_coords_comsol.ravel()), 
                temp_points.ravel(), 
                (r_grid_m, z_grid_m), 
                method='nearest'
            ).astype(np.float32)
            
            temp_map = np.nan_to_num(temp_map, nan=300.0)
            
            return temp_map, float(net_rad)
            
        except Exception as e:
            print(f"【拦截】显式几何重构或求解失败: {e}")
            raise e  

    def step(self, action):
        remove_coord, add_coord = self._decode_voxel_swap_action(action)
        
        if self._is_valid_swap(remove_coord, add_coord):
            self.current_voxel_grid[remove_coord] = 0 
            self.current_voxel_grid[add_coord] = 1    
            
        try:
            temp_map, net_rad = self._run_comsol_simulation_2d(self.current_voxel_grid)
        except Exception as e:
            return self._get_obs(), -1000.0, True, False, {"error": "COMSOL Non-convergence"}
            
        evap_map = self._calculate_evaporation_rate(temp_map) 
        feature_clusters, coords, cluster_labels = self.extract_characteristic_length(self.current_voxel_grid)
        
        if not feature_clusters:
            return self._get_obs(), -100.0, True, False, {"error": "No solid structure left"}

        cluster_lifespans = []
        for cluster_id, L_k0 in feature_clusters.items():
            v_e_k = self._get_mapped_evaporation(evap_map, coords, cluster_labels, cluster_id)
            t_fail_k = (0.2 * (L_k0 ** 2) * v_e_k) / self.rho_density
            cluster_lifespans.append(t_fail_k)
            
        topo_life = np.min(cluster_lifespans)

        reward = self._calculate_reward(net_rad, topo_life)
        obs = self._get_obs(temp_map, evap_map)
        done = False 
        
        return obs, reward, done, False, {"life": topo_life, "radiation": net_rad}
        
    def _calculate_reward(self, net_rad, topo_life):
        w1 = 10.0
        # 截断辐射增益，防止极端动作导致的数值溢出
        r_rad = w1 * np.clip((net_rad / self.initial_radiation), -5.0, 5.0)
        
        w2 = 1.0
        p_life = 0.0
        red_line = 0.3 * self.initial_life
        
        if topo_life < red_line:
            # 防梯度爆炸：将原本的指数惩罚改为平滑的归一化二次惩罚
            shortfall_ratio = (red_line - topo_life) / red_line
            p_life = -50.0 * (shortfall_ratio ** 2)
            
        return float(r_rad + w2 * p_life)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_voxel_grid = self._initialize_cylinder()
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}
        
    def _decode_voxel_swap_action(self, action):
        return tuple(), tuple() 
    def _is_valid_swap(self, rm_c, add_c):
        return False 
    def _get_mapped_evaporation(self, evap_map, coords, cluster_labels, cluster_id):
        return 1e-4 
    def _get_obs(self, temp=None, evap=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32)

if __name__ == "__main__":
    env = TungstenTopologyEnv()
    policy_kwargs = dict(normalize_images=False)
    
    model = PPO(
        "CnnPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        n_steps=128, 
        batch_size=32, 
        verbose=1, 
        tensorboard_log="./tungsten_ppo_tensorboard/"
    )
    
    print("开始强化学习训练...")
    model.learn(total_timesteps=500)
    model.save("ppo_tungsten_optimized")