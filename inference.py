import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pickle
import joblib
import logging
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class CarConfig:
    steering_gear_ratio: float
    wheel_base: float
    base_to_fa: float
    base_to_ra: float
    inertial_z: float
    mass: float
    front_corner_stiffness: float
    rear_corner_stiffness: float


class Gen4CarConfig(CarConfig):
    steering_gear_ratio: float = 15.6
    wheel_base: float = 2.755
    base_to_fa: float = 1.3535
    base_to_ra: float = 1.4015
    inertial_z: float = 3057.6
    mass: float = 2273.9
    front_corner_stiffness: float = 54920
    rear_corner_stiffness: float = 62190


class SimpleBicycleModel:
    """简化的自行车动力学模型"""
    
    def __init__(self):
        self.l_f = Gen4CarConfig.base_to_fa
        self.l_r = Gen4CarConfig.base_to_ra
        self.wheel_base = Gen4CarConfig.wheel_base
        self.steering_gear_ratio = Gen4CarConfig.steering_gear_ratio
        self.mass = Gen4CarConfig.mass
        self.c_af = Gen4CarConfig.front_corner_stiffness
        self.c_ar = Gen4CarConfig.rear_corner_stiffness
        self.i_z = Gen4CarConfig.inertial_z
    
    def simulate(self, state, control, dt=0.01):
        """
        简化的状态更新
        state: [vlon, vlat, yaw, vyaw]
        control: [acc, angle]
        """
        vlon, vlat, yaw, vyaw = state
        acc, angle = control
        
        # 前轮转角（考虑转向比）
        delta = angle / self.steering_gear_ratio
        
        # 简化的动力学更新
        # 纵向速度更新
        new_vlon = vlon + acc * dt
        
        # 侧向力计算（简化）
        beta = np.arctan2(vlat, max(vlon, 0.1))  # 侧滑角
        r = vyaw  # 横摆角速度
        
        # 侧向加速度（简化模型）
        v = np.sqrt(vlon**2 + vlat**2) if vlon != 0 else 0.1
        if v > 0:
            yaw_rate = (vlon * np.tan(delta)) / self.wheel_base
            new_vyaw = yaw_rate
        else:
            new_vyaw = vyaw
        
        # 侧向速度更新
        new_vlat = vlat + (vlon * new_vyaw) * dt
        
        # 偏航角更新
        new_yaw = yaw + vyaw * dt
        
        # 角度归一化
        new_yaw = (new_yaw + np.pi) % (2 * np.pi) - np.pi
        
        return np.array([new_vlon, new_vlat, new_yaw, new_vyaw])


class InferModel:
    def __init__(self, model, state_scaler, diff_scaler, control_scaler, base_scaler, phy_model,
                 visualizer=None, log_dir='./logs', device=None):
        params = {
            "l_f": Gen4CarConfig.base_to_fa,
            "l_r": Gen4CarConfig.base_to_ra,
            "steering_gear_ratio": Gen4CarConfig.steering_gear_ratio,
            "c_af": Gen4CarConfig.front_corner_stiffness,
            "c_ar": Gen4CarConfig.rear_corner_stiffness,
            "i_z": Gen4CarConfig.inertial_z,
            "m": Gen4CarConfig.mass,
        }
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = model.to(self.device)
        self.model.eval()
        
        self.state_scaler = state_scaler
        self.diff_scaler = diff_scaler
        self.control_scaler = control_scaler
        self.base_scaler = base_scaler
        self.phy_model = phy_model
        self.visualizer = visualizer
        
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'inference.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def denormalize_state(self, state_norm):
        """反归一化状态数据"""
        mean = torch.FloatTensor(self.state_scaler.mean_).to(state_norm.device)
        scale = torch.FloatTensor(self.state_scaler.scale_).to(state_norm.device)
        
        if state_norm.ndim == 3:
            return state_norm * scale.unsqueeze(0).unsqueeze(0) + mean.unsqueeze(0).unsqueeze(0)
        else:
            return state_norm * scale + mean

    def normalize_angle_sequence(self, angles):
        if len(angles.shape) == 1:
            angles = angles.reshape(-1, 1)
        angle_norm = angles % (2 * np.pi)
        angle_norm = np.where(angle_norm > np.pi, angle_norm - 2 * np.pi, angle_norm)
        angle_norm = np.where(angle_norm <= -np.pi, angle_norm + 2 * np.pi, angle_norm)
        diff = np.diff(angle_norm, axis=0)
        diff = np.where(diff > np.pi, diff - 2 * np.pi, diff)
        diff = np.where(diff <= -np.pi, diff + 2 * np.pi, diff)
        angle_norm[1:] = angle_norm[0] + np.cumsum(diff, axis=0)
        return angle_norm

    def read_real_and_dynamic_data(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        [model_states_real, model_states_base, control_input] = data[1]
        # 只使用前4维状态（vlon, vlat, yaw, vyaw），与训练时一致
        state_data = model_states_real[:, [2, 3, 4, 5]]   # vlon, vlat, yaw, vyaw
        base_data = model_states_base[:, [2, 3, 4, 5]]     # 物理模型输出的前4维
        control_data = control_input[:, :]      # acc, angle
        xy_data = model_states_real[:, :2]      # x, y
        return np.array(state_data), np.array(control_data), np.array(base_data), np.array(xy_data)

    def compute_xy_trajectory(self, states, start_pos, dt=0.01):
        """根据状态序列计算XY轨迹"""
        trajectory = [start_pos.copy()]
        current_pos = start_pos.copy()
        
        for state in states:
            vlon = state[0]  # 纵向速度
            vlat = state[1]  # 横向速度
            yaw = state[2]   # 偏航角
            
            # 将速度从车体坐标系转换到全局坐标系
            vx = vlon * np.cos(yaw) - vlat * np.sin(yaw)
            vy = vlon * np.sin(yaw) + vlat * np.cos(yaw)
            
            # 更新位置
            current_pos[0] += vx * dt
            current_pos[1] += vy * dt
            
            trajectory.append(current_pos.copy())
        
        return np.array(trajectory)
    
    def predict_net_single_step(self, hist_state, hist_control, base_state, cfg_tensor=None):
        # 保存原始形状
        orig_shape_hist = hist_state.shape
        orig_shape_ctrl = hist_control.shape
        orig_shape_base = base_state.shape
        
        # 归一化处理
        hist_state_norm = self.state_scaler.transform(hist_state.reshape(-1, orig_shape_hist[-1])).reshape(orig_shape_hist)
        hist_control_norm = self.control_scaler.transform(hist_control.reshape(-1, orig_shape_ctrl[-1])).reshape(orig_shape_ctrl)
        base_state_norm = self.base_scaler.transform(base_state.reshape(-1, orig_shape_base[-1])).reshape(orig_shape_base)

        # 转换为Tensor
        hist_state_tensor = torch.FloatTensor(hist_state_norm).unsqueeze(0).to(self.device)
        hist_control_tensor = torch.FloatTensor(hist_control_norm).unsqueeze(0).to(self.device)
        base_state_tensor = torch.FloatTensor(base_state_norm).unsqueeze(0).to(self.device)
        cfg_tensor = cfg_tensor.to(self.device) if cfg_tensor is not None else None

        with torch.no_grad():
            delta_norm, corrected_norm = self.model(
                hist_state_tensor, hist_control_tensor, base_state_tensor, cfg_tensor
            )

        corrected_norm = corrected_norm.squeeze(0).squeeze(0).cpu().numpy()
        delta_norm = delta_norm.squeeze(0).cpu().numpy()

        # 反归一化
        corrected = self.state_scaler.inverse_transform(corrected_norm.reshape(1, -1)).flatten()
        delta = self.diff_scaler.inverse_transform(delta_norm.reshape(1, -1)).flatten()
        base_state = self.base_scaler.inverse_transform(base_state_norm.reshape(1, -1)).flatten()

        return corrected, delta, base_state

    def autoregressive_predict(self, initial_state, controls, pred_len=100, dt=0.01, inference_mask=[1, 1, 1, 1]):
        current_state = initial_state.copy()
        hybrid_states = []
        physics_states = []
        delta_list = []

        mask = np.array(inference_mask)

        for t in range(pred_len):
            control = controls[t] if t < len(controls) else controls[-1]

            # 物理模型预测
            physics_next = self.phy_model.simulate(current_state, control)
            physics_states.append(physics_next.copy())

            # 神经网络预测（使用当前状态和控制作为历史）
            hist_state = np.array([current_state])
            hist_control = np.array([control])
            base_state = np.array([physics_next])
            cfg_tensor = torch.FloatTensor([1, 1, 0])

            corrected, delta, base = self.predict_net_single_step(hist_state, hist_control, base_state, cfg_tensor)
            
            # 应用mask
            hybrid_next = physics_next.copy()
            hybrid_next[:len(mask)] = mask * corrected[:len(mask)] + (1 - mask) * physics_next[:len(mask)]
            
            delta_list.append(delta)
            hybrid_states.append(hybrid_next.copy())
            current_state = hybrid_next.copy()

        return np.array(hybrid_states), np.array(physics_states), np.array(delta_list)

    def run_physics_only_full(self, state_data, control_data, xy_data):
        """运行纯物理模型获取完整状态序列和XY轨迹"""
        n_samples = len(state_data)
        physics_states = []
        physics_xy = []
        
        if self.phy_model is not None:
            current_state = state_data[0].copy()
            current_xy = xy_data[0].copy()
            
            physics_states.append(current_state.copy())
            physics_xy.append(current_xy.copy())
            
            for t in range(1, n_samples):
                control = control_data[t-1]
                physics_next = self.phy_model.simulate(current_state, control)
                
                # 计算XY轨迹
                vlon = physics_next[0]
                vlat = physics_next[1]
                yaw = physics_next[2]
                vx = vlon * np.cos(yaw) - vlat * np.sin(yaw)
                vy = vlon * np.sin(yaw) + vlat * np.cos(yaw)
                current_xy[0] += vx * 0.01
                current_xy[1] += vy * 0.01
                
                physics_states.append(physics_next.copy())
                physics_xy.append(current_xy.copy())
                current_state = physics_next.copy()
        else:
            physics_states = base_data
            physics_xy = xy_data
        
        return np.array(physics_states), np.array(physics_xy)

    def run_inference_on_file(self, file_path, output_dir='./inference_results'):
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取数据
        state_data, control_data, base_data, xy_data = self.read_real_and_dynamic_data(file_path)
        n_samples = len(state_data)
        
        # 运行纯物理模型获取完整序列
        physics_preds_full, physics_xy_full = self.run_physics_only_full(state_data, control_data, xy_data)
        
        # 使用混合模型预测（自回归方式）
        hybrid_pred = []
        hybrid_xy = []
        
        # 使用物理模型的输出作为初始状态
        current_hist_state = physics_preds_full[:self.model.hist_len]
        current_xy = physics_xy_full[self.model.hist_len-1].copy()
        
        for i in range(self.model.hist_len, n_samples):
            hist_state = current_hist_state[-self.model.hist_len:]
            hist_control = control_data[i-self.model.hist_len:i]
            base_state = physics_preds_full[i-1:i]
            
            # cfg_tensor 需要是二维的 (batch_size, 3)
            cfg_tensor = torch.FloatTensor([[1, 1, 0]])
            corrected, delta, base = self.predict_net_single_step(hist_state, hist_control, base_state, cfg_tensor)
            
            hybrid_pred.append(corrected)
            
            # 计算XY轨迹
            vlon = corrected[0]
            vlat = corrected[1]
            yaw = corrected[2]
            vx = vlon * np.cos(yaw) - vlat * np.sin(yaw)
            vy = vlon * np.sin(yaw) + vlat * np.cos(yaw)
            current_xy[0] += vx * 0.01
            current_xy[1] += vy * 0.01
            hybrid_xy.append(current_xy.copy())
            
            # 更新历史状态
            current_hist_state = np.vstack([current_hist_state[1:], corrected.reshape(1, -1)])
        
        hybrid_pred = np.array(hybrid_pred)
        hybrid_xy = np.array(hybrid_xy)
        
        # 截取对应的真实值和物理预测值（从hist_len开始）
        compare_start = self.model.hist_len
        state_data_compare = state_data[compare_start:]
        physics_pred_compare = physics_preds_full[compare_start:]
        
        # 计算误差
        rmse_base = np.sqrt(np.mean((state_data_compare - physics_pred_compare)**2, axis=0))
        rmse_hybrid = np.sqrt(np.mean((state_data_compare - hybrid_pred)**2, axis=0))
        
        self.logger.info(f"RMSE - Baseline: {rmse_base}")
        self.logger.info(f"RMSE - Hybrid: {rmse_hybrid}")
        self.logger.info(f"Improvement: {rmse_base - rmse_hybrid}")
        
        # 创建完整的预测序列（前部分使用物理模型，后部分使用混合模型）
        hybrid_preds_full = np.copy(physics_preds_full)
        hybrid_xy_full = np.copy(physics_xy_full)
        if len(hybrid_pred) > 0:
            hybrid_preds_full[compare_start:compare_start+len(hybrid_pred)] = hybrid_pred
            hybrid_xy_full[compare_start:compare_start+len(hybrid_xy)] = hybrid_xy
        
        # 保存结果
        results = {
            'state_data': state_data_compare,
            'control_data': control_data,
            'base_data': base_data,
            'physics_pred': physics_pred_compare,
            'hybrid_pred': hybrid_pred,
            'xy_data': xy_data,
            'physics_xy_full': physics_xy_full,
            'hybrid_xy_full': hybrid_xy_full,
            'physics_preds_full': physics_preds_full,
            'hybrid_preds_full': hybrid_preds_full,
            'rmse_base': rmse_base,
            'rmse_hybrid': rmse_hybrid,
            'start_idx': 0,
            'hist_len': self.model.hist_len
        }
        
        file_name = os.path.basename(file_path)
        with open(os.path.join(output_dir, f'inference_result_{file_name}'), 'wb') as f:
            pickle.dump(results, f)
        
        # 可视化
        if self.visualizer is not None:
            trip_id = os.path.splitext(file_name)[0]
            method_save_dir = os.path.join(output_dir, trip_id)
            os.makedirs(method_save_dir, exist_ok=True)
            
            # 状态对比图
            self.visualizer.visualize_model_comparison(
                physics_preds_full, hybrid_preds_full, state_data,
                start_idx=0, hist_len=self.model.hist_len,
                save_path=os.path.join(method_save_dir, 'state_comparison.png')
            )
            
            # XY轨迹对比图
            self.visualizer.plot_xy_comparison(
                xy_data, physics_xy_full, hybrid_xy_full,
                save_path=os.path.join(method_save_dir, 'xy_comparison.png'),
                title=f'Trajectory Comparison - {trip_id}'
            )
            
            # 累积误差曲线
            cumulative_errors = hybrid_pred - state_data_compare
            self.visualizer.plot_error_curves(
                cumulative_errors, start_idx=0, hist_len=self.model.hist_len,
                save_path=os.path.join(method_save_dir, 'error_curves.png')
            )
            
            # 400帧后波动分析
            if n_samples > 400:
                self.visualizer.plot_fluctuation_analysis(
                    state_data_compare, physics_pred_compare, hybrid_pred,
                    threshold_frame=400-self.model.hist_len,
                    save_path=os.path.join(method_save_dir, 'fluctuation_analysis.png')
                )
                
                self.visualizer.plot_fluctuation_detailed(
                    state_data_compare, physics_pred_compare, hybrid_pred,
                    start_frame=400-self.model.hist_len,
                    window_size=200,
                    save_path=os.path.join(method_save_dir, 'fluctuation_detail.png')
                )
            
            # 数据对比可视化
            self.visualizer.visualize_data_comparison(
                state_data, base_data, control_data,
                save_path=method_save_dir
            )
        
        return results

    def batch_inference(self, data_dir, output_dir='./inference_results', n_files=0):
        os.makedirs(output_dir, exist_ok=True)
        
        files = glob.glob(os.path.join(data_dir, '*.pkl'))
        self.logger.info(f"Found {len(files)} files for inference")
        
        # 如果指定了文件数量，只使用前n_files个文件
        if n_files > 0 and n_files < len(files):
            files = files[:n_files]
            self.logger.info(f"Using first {n_files} files for inference")
        
        all_rmse_base = []
        all_rmse_hybrid = []
        
        for file_path in files:
            self.logger.info(f"Processing: {os.path.basename(file_path)}")
            results = self.run_inference_on_file(file_path, output_dir)
            all_rmse_base.append(results['rmse_base'])
            all_rmse_hybrid.append(results['rmse_hybrid'])
        
        # 计算平均误差
        mean_rmse_base = np.mean(all_rmse_base, axis=0)
        mean_rmse_hybrid = np.mean(all_rmse_hybrid, axis=0)
        
        self.logger.info("\n=== Batch Inference Summary ===")
        self.logger.info(f"Mean RMSE - Baseline: {mean_rmse_base}")
        self.logger.info(f"Mean RMSE - Hybrid: {mean_rmse_hybrid}")
        self.logger.info(f"Mean Improvement: {mean_rmse_base - mean_rmse_hybrid}")
        
        return mean_rmse_base, mean_rmse_hybrid