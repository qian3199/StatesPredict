import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pickle
import joblib
from scipy import signal
from scipy.signal import butter, filtfilt
import sys
import argparse
import time
from datetime import datetime
import json

def butter_lowpass_filter(data, cutoff_freq, fs=100, order=1):
    """低通滤波器 - 与训练数据生成时使用的一致"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

from model import DyTR_LSTM
from InferVisualizer import InferVisualizer


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


class State(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

class ControlInput(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj


class LinearBicycleModel:
    """
    线性自行车动力学模型 (Linear Dynamic Bicycle Model) - 与训练数据生成时使用的模型一致
    状态: [x, y, vlon, vlat, yaw, omega]
    控制: [acc, steering_angle]
    """
    States = ['x', 'y', 'vlon', 'vlat', 'yaw', 'omega']
    ControlInputs = ['acc', 'steering_angle']

    def __init__(self, parameters, steer_delay_steps=2, process_noise_std=0.005):
        self.m = parameters.get('m', 2273.9)
        self.Iz = parameters.get('i_z', 3057.6)
        self.lf = parameters.get('l_f', 1.3535)
        self.lr = parameters.get('l_r', 1.4015)
        self.Caf = parameters.get('c_af', 54920)
        self.Car = parameters.get('c_ar', 62190)
        self.ratio = parameters.get('steering_gear_ratio', 15.6)
        # 添加训练数据生成时的延迟和噪声
        self.steer_delay_steps = steer_delay_steps
        self.steer_history = []
        self.process_noise_std = process_noise_std

    def forward(self, state, control, dt=0.01):
        """线性动力学模型更新 - 添加延迟和噪声"""
        acc, steering_angle = control

        # 转向延迟
        self.steer_history.append(steering_angle)
        if len(self.steer_history) > self.steer_delay_steps:
            self.steer_history.pop(0)
        delayed_steer = np.mean(self.steer_history) if len(self.steer_history) > 0 else steering_angle

        new_control = np.array([acc, delayed_steer])

        # 过程噪声
        noise = np.random.normal(0, self.process_noise_std, 6)
        noisy_state = state + noise

        x, y, vlon, vlat, yaw, omega = noisy_state
        acc_noisy, delta = new_control
        delta = delta / self.ratio

        beta = np.arctan2(vlat, vlon) if abs(vlon) > 0.01 else 0

        alpha_f = delta - np.arctan2(self.lf * omega + vlat, vlon) if abs(vlon) > 0.01 else 0
        alpha_r = -np.arctan2(self.lr * omega - vlat, vlon) if abs(vlon) > 0.01 else 0

        Fyf = self.Caf * alpha_f
        Fyr = self.Car * alpha_r

        d_vlon = (Fyf * np.sin(delta) + self.m * vlat * omega + Fyr * np.sin(beta) - Fyf * np.cos(delta)) / self.m + acc_noisy
        d_vlat = (-Fyf * np.cos(delta) - Fyr + self.m * vlon * omega) / self.m
        d_omega = (Fyf * self.lf * np.cos(delta) - Fyr * self.lr) / self.Iz

        vlon_new = vlon + d_vlon * dt
        vlat_new = vlat + d_vlat * dt
        omega_new = omega + d_omega * dt

        yaw_new = yaw + omega_new * dt

        v_total = np.sqrt(vlon_new**2 + vlat_new**2)
        if v_total > 0.01:
            x_new = x + v_total * np.cos(yaw_new + np.arctan2(vlat_new, vlon_new)) * dt
            y_new = y + v_total * np.sin(yaw_new + np.arctan2(vlat_new, vlon_new)) * dt
        else:
            x_new = x + vlon_new * np.cos(yaw_new) * dt
            y_new = y + vlon_new * np.sin(yaw_new) * dt

        return np.array([x_new, y_new, vlon_new, vlat_new, yaw_new, omega_new])

    def simulate(self, initial_state, control_input, n=1, dt=0.01):
        """
        使用动力学模型积分 n 步
        initial_state: (6,) 或 (6,1)
        control_input: (2,) 或 (2,1)
        return: traj (6, n+1), success
        """
        if initial_state.ndim == 1:
            init = initial_state.reshape(-1, 1)
        else:
            init = initial_state
        if control_input.ndim == 1:
            ctrl = control_input.reshape(-1, 1)
        else:
            ctrl = control_input

        traj = np.zeros((6, n+1))
        traj[:, 0] = init.flatten()

        for step in range(n):
            cur = traj[:, step]
            a = ctrl[0, 0] if ctrl.shape[1] == 1 else ctrl[0, step]
            delta = ctrl[1, 0] if ctrl.shape[1] == 1 else ctrl[1, step]
            control = np.array([a, delta])

            next_state = self.forward(cur, control, dt)
            traj[:, step+1] = next_state

        return traj, True


class InferModel:
    def __init__(self, model_config, model_flag, model_path, scaler_path, dt, feature_name, control_name, device='cpu'):
        params = {
            "l_f": Gen4CarConfig.base_to_fa,
            "l_r": Gen4CarConfig.base_to_ra,
            "steering_gear_ratio": Gen4CarConfig.steering_gear_ratio,
            "c_af": Gen4CarConfig.front_corner_stiffness,
            "c_ar": Gen4CarConfig.rear_corner_stiffness,
            "i_z": Gen4CarConfig.inertial_z,
            "m": Gen4CarConfig.mass,
        }
        
        self.device = device
        self.model_config = model_config
        self.dt = dt
        self.feature_name = feature_name
        self.control_name = control_name
        self.model_flag = model_flag
        phy_params = {
            'm': Gen4CarConfig.mass,
            'i_z': Gen4CarConfig.inertial_z,
            'l_f': Gen4CarConfig.base_to_fa,
            'l_r': Gen4CarConfig.base_to_ra,
            'c_af': Gen4CarConfig.front_corner_stiffness,
            'c_ar': Gen4CarConfig.rear_corner_stiffness,
            'steering_gear_ratio': Gen4CarConfig.steering_gear_ratio
        }
        self.phy_model = LinearBicycleModel(phy_params)
        net_model = self.load_trained_model(model_path, model_config, device)
        self.model = net_model.to(device)
        self.model.eval()

        self.visualizer = InferVisualizer(
            feature_name=feature_name, 
            control_name=control_name, 
            dt=dt
        )

        self.scalers = None
        if scaler_path is not None and os.path.exists(scaler_path):
            self.scalers = joblib.load(scaler_path)
            print("Scaler 加载成功，包含的键:", self.scalers.keys())
        elif scaler_path is not None:
            print(f"Warning: Scaler file {scaler_path} not found, proceeding without scaling")

    def read_real_and_dynamic_data(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            state_data = data.get('state', data.get('model_states_real', np.zeros((100, 7))))
            control_data = data.get('control', data.get('control_input', np.zeros((100, 2))))
            xy_data = data.get('xy', data.get('xy_data', np.zeros((100, 2))))
            base_data = data.get('base', np.zeros_like(state_data))
            
            if state_data.ndim == 2 and state_data.shape[1] >= 7:
                state_data = state_data[:, :7]
            elif state_data.ndim == 2 and state_data.shape[1] == 4:
                padding = np.zeros((state_data.shape[0], 3))
                state_data = np.concatenate([state_data, padding], axis=1)
        else:
            if len(data) > 1 and isinstance(data[1], (list, tuple)):
                [model_states_real, model_states_base, control_input] = data[1]
                state_data = model_states_real[:, 2:]
                base_data = model_states_base[:, 2:]
                control_data = control_input[:, :]
                xy_data = model_states_real[:, :2]
            else:
                state_data = np.zeros((100, 7))
                control_data = np.zeros((100, 2))
                xy_data = np.zeros((100, 2))
                base_data = np.zeros((100, 7))
        
        return np.array(state_data), np.array(control_data), np.array(base_data), np.array(xy_data)

    def load_trained_model(self, model_path, model_config, device='cpu'):
        device = torch.device(device)
        try:
            checkpoint = torch.load(model_path, map_location=device)
            print(f"checkpoint keys: {list(checkpoint.keys())}")
            
            if 'model_state_dict' in checkpoint:
                model = DyTR_LSTM(**model_config)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from checkpoint: epoch={checkpoint.get('epoch', 'unknown')}")
            else:
                model = DyTR_LSTM(**model_config)
                model.load_state_dict(checkpoint)
                print(f"Loaded model directly from state dict")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            model = DyTR_LSTM(**model_config)
            print(f"Created new model with config: {model_config}")
        
        return model

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

    def unwrap_angle(self, current_angle, prev_angle):
        diff = current_angle - prev_angle
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return prev_angle + diff

    def predict_net_single_step(self, hist_state, hist_control, base_state, cfg_tensor=None):
        if self.scalers is not None:
            orig_shape_hist = hist_state.shape
            hist_state_norm = self.scalers['state'].transform(hist_state.reshape(-1, orig_shape_hist[-1])).reshape(orig_shape_hist)
            orig_shape_ctrl = hist_control.shape
            hist_control_norm = self.scalers['control'].transform(hist_control.reshape(-1, orig_shape_ctrl[-1])).reshape(orig_shape_ctrl)
            orig_shape_base = base_state.shape
            base_state_norm = self.scalers['base'].transform(base_state.reshape(-1, orig_shape_base[-1])).reshape(orig_shape_base)
        else:
            hist_state_norm, hist_control_norm, base_state_norm = hist_state, hist_control, base_state

        hist_state_tensor = torch.FloatTensor(hist_state_norm).unsqueeze(0).to(self.device)
        hist_control_tensor = torch.FloatTensor(hist_control_norm).unsqueeze(0).to(self.device)
        
        base_state_tensor = torch.FloatTensor(base_state_norm).unsqueeze(0).to(self.device)
        if base_state_tensor.dim() == 2:
            base_state_tensor = base_state_tensor.unsqueeze(1)
        
        if cfg_tensor is not None:
            cfg_tensor = cfg_tensor.to(self.device)
            if cfg_tensor.size(-1) == 1:
                cfg_tensor = torch.cat([cfg_tensor, cfg_tensor, cfg_tensor], dim=-1)
        else:
            cfg_tensor = torch.FloatTensor([[2237.9, 15, 10]]).to(self.device)

        with torch.no_grad():
            delta_norm, corrected_norm = self.model(
                hist_state_tensor,         
                hist_control_tensor,      
                base_state_tensor,
                cfg_tensor
            )
        
        corrected_norm = corrected_norm.squeeze(0).squeeze(0).cpu().numpy()
        delta_norm = delta_norm.squeeze(0).squeeze(0).cpu().numpy()
        
        if delta_norm.ndim == 0:
            delta_norm = np.array([delta_norm])
        if corrected_norm.ndim == 0:
            corrected_norm = np.array([corrected_norm])
        
        if len(delta_norm) == 1 and self.model_config.get('state_dim', 4) == 4:
            delta_norm = np.array([delta_norm[0], delta_norm[0], delta_norm[0], delta_norm[0]])
        if len(corrected_norm) == 1 and self.model_config.get('state_dim', 4) == 4:
            corrected_norm = np.array([corrected_norm[0], corrected_norm[0], corrected_norm[0], corrected_norm[0]])

        if self.scalers is not None:
            corrected = self.scalers['state'].inverse_transform(corrected_norm.reshape(1, -1)).flatten()
            delta = self.scalers['diff'].inverse_transform(delta_norm.reshape(1, -1)).flatten()
            base_state = self.scalers['base'].inverse_transform(base_state_norm.reshape(1, -1)).flatten()
        else:
            corrected, delta = corrected_norm, delta_norm
            base_state = base_state_norm.flatten() if hasattr(base_state_norm, 'flatten') else np.array(base_state_norm).flatten()

        delta = np.array(delta).flatten()
        corrected = np.array(corrected).flatten()
        base_state = np.array(base_state).flatten()

        if len(delta) < 4:
            delta = np.pad(delta, (0, 4 - len(delta)), mode='constant')
        if len(corrected) < 4:
            corrected = np.pad(corrected, (0, 4 - len(corrected)), mode='constant')
        if len(base_state) < 4:
            base_state = np.pad(base_state, (0, 4 - len(base_state)), mode='constant')

        return corrected, delta, base_state


    def predict_physics_single_step(self, current_states, current_control, xy_data_t):
        """使用线性动力学模型进行单步预测"""
        # 提取当前状态和控制
        vlon, vlat, yaw, vyaw = current_states[-1, :4]
        x, y = xy_data_t[-1, 0], xy_data_t[-1, 1]
        acc, angle = current_control[-1, :2]
        
        # 构造动力学模型的状态: [x, y, vlon, vlat, yaw, omega]
        state_6d = np.array([x, y, vlon, vlat, yaw, vyaw])
        control = np.array([acc, angle])
        
        # 调用动力学模型模拟一步
        traj_6d, success = self.phy_model.simulate(state_6d, control, n=1, dt=self.dt)
        
        # 提取下一时刻的状态
        next_state_6d = traj_6d[:, 1]
        next_x, next_y, next_vlon, next_vlat, next_yaw, next_vyaw = next_state_6d
        
        # 计算加速度
        acc_lon = (next_vlon - vlon) / self.dt
        acc_lat = (next_vlat - vlat) / self.dt
        acc_yaw = (next_vyaw - vyaw) / self.dt
        
        # 组合输出 (4D状态 + 3D加速度)
        physics_next = np.array([next_vlon, next_vlat, next_yaw, next_vyaw])
        xy_data_t = np.array([[next_x, next_y]])
        phy_pred = np.concatenate([
            physics_next,
            np.array([acc_lon, acc_lat, acc_yaw])
        ], axis=0)
        phy_pred = phy_pred.reshape(1, -1)
        return phy_pred, xy_data_t

    def run_physics_only_full(self, state_data, control_data, xy_data):
        n_samples = len(state_data)
        physics_states = []
        physics_xy = []
        
        if state_data.shape[1] >= 4:
            current_hist_state = state_data[0:1, :]
        else:
            padding = np.zeros((1, 7 - state_data.shape[1]))
            current_hist_state = np.concatenate([state_data[0:1, :], padding], axis=1)
        
        current_hist_control = control_data[0:1, :] if len(control_data) > 0 else np.zeros((1, 2))
        xy_data_t = xy_data[0:1, :] if len(xy_data) > 0 else np.zeros((1, 2))

        physics_states.append(current_hist_state[0])
        physics_xy.append(xy_data_t[0])

        for t in range(1, n_samples):
            physics_pred, xy_data_phy = self.predict_physics_single_step(current_hist_state, current_hist_control, xy_data_t)
            
            if t < n_samples:
                if t < len(control_data):
                    current_hist_control = np.vstack([current_hist_control[1:, :], control_data[t:t+1, :]])
                xy_data_t = xy_data_phy
                current_hist_state = np.vstack([current_hist_state[1:, :], physics_pred])
            
            physics_states.append(physics_pred[0])
            physics_xy.append(xy_data_phy[0])

        physics_states = np.array(physics_states)
        if physics_states.shape[1] > 2:
            tmp_norm_angles = self.normalize_angle_sequence(physics_states[:, 2])
            physics_states[:, 2] = tmp_norm_angles.flatten()
        
        # 添加低通滤波 - 与训练数据生成时保持一致
        fs = 1.0 / self.dt  # 采样频率
        cutoff_freq = 0.01   # 截止频率
        # physics_states 的格式: [vlon, vlat, yaw, omega, acc_lon, acc_lat, acc_yaw]
        if physics_states.shape[1] >= 4:
            # 对状态量进行滤波 (vlon, vlat, yaw, omega)
            physics_states[:, 0] = butter_lowpass_filter(physics_states[:, 0], cutoff_freq, fs, order=1)
            physics_states[:, 1] = butter_lowpass_filter(physics_states[:, 1], cutoff_freq, fs, order=1)
            physics_states[:, 2] = butter_lowpass_filter(physics_states[:, 2], cutoff_freq, fs, order=1)
            physics_states[:, 3] = butter_lowpass_filter(physics_states[:, 3], cutoff_freq, fs, order=1)
            # 对加速度进行滤波
            if physics_states.shape[1] >= 7:
                physics_states[:, 4] = butter_lowpass_filter(physics_states[:, 4], cutoff_freq, fs, order=1)
                physics_states[:, 5] = butter_lowpass_filter(physics_states[:, 5], cutoff_freq, fs, order=1)
                physics_states[:, 6] = butter_lowpass_filter(physics_states[:, 6], cutoff_freq, fs, order=1)
        
        return physics_states, np.array(physics_xy)

    def autoregressive_predict(self, state_data, control_data, base_data, xy_data,
                               start_idx, hist_len=15, cfg_data=None):
        n_samples = len(state_data)
        predictions, xy_predict, deltas, targets, cumulative_errors = [], [], [], [], []

        physics_states, physics_xy = self.run_physics_only_full(
            state_data[:start_idx+hist_len], control_data[:start_idx+hist_len], xy_data[:start_idx+hist_len]
        )

        current_hist_state = physics_states[start_idx:start_idx+hist_len, :]
        current_hist_control = control_data[start_idx:start_idx+hist_len, :] if len(control_data) > start_idx+hist_len else np.zeros((hist_len, 2))
        xy_data_t = physics_xy[start_idx+hist_len-1:start_idx+hist_len, :]
        prev_yaw = current_hist_state[-1, 2]

        for t in range(start_idx+hist_len, n_samples):
            physics_pred, xy_data_phy = self.predict_physics_single_step(current_hist_state, current_hist_control, xy_data_t)
            physics_yaw = physics_pred[0, 2]
            physics_yaw_cont = self.unwrap_angle(physics_yaw, prev_yaw)
            physics_pred[0, 2] = physics_yaw_cont
            
            if cfg_data is not None:
                current_cfg = cfg_data[t:t+1]
                correct, delta, base = self.predict_net_single_step(
                    hist_state=current_hist_state[:, :4], 
                    hist_control=current_hist_control,
                    base_state=physics_pred[:, :4],
                    cfg_tensor=current_cfg
                )
                net_v_lat = delta[1]  
                net_v_lon = net_v_yaw = net_acc_lon = net_acc_lat = net_acc_yaw = 0
                net_yaw = 0
                yaw_corrected = physics_pred[0, 2] + net_yaw
                yaw_corrected = self.unwrap_angle(yaw_corrected, prev_yaw)
                net_vx = net_v_lon * np.cos(yaw_corrected) - net_v_lat * np.sin(yaw_corrected)
                net_vy = net_v_lon * np.sin(yaw_corrected) + net_v_lat * np.cos(yaw_corrected)
                delta_x = net_vx * self.dt
                delta_y = net_vy * self.dt
                data_x = xy_data_phy[0, 0] + delta_x
                data_y = xy_data_phy[0, 1] + delta_y
                physics_pred[0, 1] += net_v_lat
                physics_pred[0, 2] = yaw_corrected
            else:
                data_x, data_y = xy_data_phy[0, 0], xy_data_phy[0, 1]
                net_v_lon = net_v_lat = net_yaw = net_v_yaw = net_acc_lon = net_acc_lat = net_acc_yaw = 0
                
            delta_data = [net_v_lon, net_v_lat, net_yaw, net_v_yaw, net_acc_lon, net_acc_lat, net_acc_yaw]
            if t + 1 < n_samples:
                if t < len(control_data):
                    current_hist_control = np.vstack([current_hist_control[1:, :], control_data[t:t+1, :]])
                xy_data_t = np.array([[data_x, data_y]])
                current_hist_state = np.vstack([current_hist_state[1:, :], physics_pred])
            prev_yaw = physics_pred[0, 2]
            xy_predict.append([data_x, data_y])
            predictions.append(physics_pred[0])
            deltas.append(delta_data)
            targets.append(state_data[t] if t < len(state_data) else np.zeros(7))
            cumulative_errors.append(physics_pred[0] - (state_data[t] if t < len(state_data) else np.zeros(7)))
            
        return np.array(predictions), np.array(deltas), np.array(targets), np.array(cumulative_errors), np.array(xy_predict)

    def autoregressive_predict_ind(self, state_data, control_data, base_data, xy_data,
                           start_idx, hist_len=15, cfg_data=None):
        n_samples = len(state_data)
        print("运行纯物理模型获取完整序列...")
        physics_preds_full, physics_xy_full = self.run_physics_only_full(state_data, control_data, xy_data)
        
        predictions, xy_predict, deltas, targets, cumulative_errors = [], [], [], [], []
        compare_start = start_idx + hist_len
        print(f"网络单独迭代并相加 (从t={compare_start}开始)")
        
        for t in range(compare_start, n_samples):
            print(f"混合模型预测: {t}/{n_samples - 1}", end='\r')
            pure_physics_pred = physics_preds_full[t:t+1, :]
            pure_xy_phy = physics_xy_full[t:t+1, :]
            hist_state = physics_preds_full[t-hist_len:t, :]
            hist_control = control_data[t-hist_len:t, :] if t-hist_len >= 0 and t < len(control_data) else np.zeros((hist_len, 2))

            if cfg_data is not None:
                current_cfg = cfg_data[t:t + 1]
                correct, delta, base = self.predict_net_single_step(
                    hist_state=hist_state[:, :4],  
                    hist_control=hist_control,
                    base_state=pure_physics_pred[:, :4],  
                    cfg_tensor=current_cfg
                )
                delta[1] = correct[1] - base[1]
                net_v_lon = net_v_lat = net_v_yaw = net_yaw = net_acc_lon = net_acc_lat = net_acc_yaw = 0
                net_v_lat = delta[1]
                yaw_corrected = pure_physics_pred[0, 2] + net_yaw
                net_vx = net_v_lon * np.cos(yaw_corrected) - net_v_lat * np.sin(yaw_corrected)
                net_vy = net_v_lon * np.sin(yaw_corrected) + net_v_lat * np.cos(yaw_corrected)
                delta_x = net_vx * self.dt
                delta_y = net_vy * self.dt
                data_x = pure_xy_phy[0, 0] + delta_x
                data_y = pure_xy_phy[0, 1] + delta_y
                delta_data = [net_v_lon, net_v_lat, net_yaw, net_v_yaw, net_acc_lon, net_acc_lat, net_acc_yaw]
            else:
                data_x, data_y = pure_xy_phy[0, 0], pure_xy_phy[0, 1]
                delta_data = [0,0,0,0,0,0,0]
                
            mixed_pred = pure_physics_pred.copy()
            mixed_pred[0, 0] += net_v_lon
            mixed_pred[0, 1] += net_v_lat
            mixed_pred[0, 2] += net_yaw
            mixed_pred[0, 3] += net_v_yaw
            mixed_pred[0, 4] += net_acc_lon
            mixed_pred[0, 5] += net_acc_lat
            mixed_pred[0, 6] += net_acc_yaw
            
            xy_predict.append([data_x, data_y])
            predictions.append(mixed_pred[0]) 
            deltas.append(delta_data)
            targets.append(state_data[t] if t < len(state_data) else np.zeros(7))
            cumulative_errors.append(mixed_pred[0] - (state_data[t] if t < len(state_data) else np.zeros(7)))
            
        print("\n混合模型预测完成!")
        predictions = np.array(predictions)
        
        if predictions.size > 0 and predictions.shape[1] > 2:
            hist_yaws = physics_preds_full[:compare_start, 2].reshape(-1, 1)
            pred_yaws = predictions[:, 2].reshape(-1, 1)
            tmp_norm_angles = np.vstack([hist_yaws, pred_yaws])
            tmp_norm_angles = self.normalize_angle_sequence(tmp_norm_angles)
            predictions[:, 2] = tmp_norm_angles[compare_start:].flatten() 
        
        deltas = np.array(deltas)
        targets = np.array(targets)
        cumulative_errors = np.array(cumulative_errors)
        
        if predictions.ndim == 3 and predictions.shape[1] == 1:
            predictions = predictions.reshape(predictions.shape[0], predictions.shape[2])
        if cumulative_errors.ndim == 3 and cumulative_errors.shape[1] == 1:
            cumulative_errors = cumulative_errors.reshape(cumulative_errors.shape[0], cumulative_errors.shape[2])
            
        return predictions, deltas, targets, cumulative_errors, xy_data, np.array(xy_predict)

    def create_full_hybrid_prediction(self, physics_preds_full, hybrid_preds, start_idx, hist_len, n_total):
        compare_start = start_idx + hist_len
        full_hybrid = np.full((n_total, physics_preds_full.shape[1]), np.nan)
        full_hybrid[:compare_start] = physics_preds_full[:compare_start]
        n_hybrid = len(hybrid_preds)
        if compare_start + n_hybrid <= n_total:
            full_hybrid[compare_start:compare_start + n_hybrid] = hybrid_preds
        else:
            full_hybrid[compare_start:] = hybrid_preds[:n_total - compare_start]
        return full_hybrid


    def run_single_trip_inference(self, data_path, cfg_value=2237.9, start_idx=0, hist_len=15, save_path=None, trip_id=None):
        state_data, control_data, base_data, xy_data = self.read_real_and_dynamic_data(data_path)
        n_samples = len(state_data)
        cfg_data = torch.FloatTensor(np.ones((n_samples, 1)) * cfg_value)
        
        physics_preds_full, physics_xy_full = self.run_physics_only_full(state_data, control_data, xy_data)
        preds, deltas, targets, cum_errors, xy_preds = self.autoregressive_predict(
            state_data, control_data, base_data, xy_data, start_idx, hist_len, cfg_data
        )
        preds_add, deltas_add, targets_add, cum_errors_add, _, xy_preds_add = self.autoregressive_predict_ind(
            state_data, control_data, base_data, xy_data, start_idx, hist_len, cfg_data
        )
        
        if trip_id is None:
            trip_id = os.path.basename(data_path).replace('.pkl', '')
        compare_start = start_idx + hist_len

        def process_method(method_name, predictions, deltas, targets, cumulative_errors, xy_predict):
            hybrid_preds_full = self.create_full_hybrid_prediction(physics_preds_full, predictions, start_idx, hist_len, n_samples)
            hybrid_xy_full = np.zeros_like(physics_xy_full)
            hybrid_xy_full[:compare_start] = physics_xy_full[:compare_start]
            if len(xy_predict) > 0:
                end_idx = min(compare_start + len(xy_predict), len(hybrid_xy_full))
                hybrid_xy_full[compare_start:end_idx] = xy_predict[:end_idx - compare_start]
            else:
                hybrid_xy_full[compare_start:] = physics_xy_full[compare_start:]
                
            if save_path is not None:
                method_save_dir = os.path.join(save_path, trip_id, method_name)
                os.makedirs(method_save_dir, exist_ok=True)
                
                self.visualizer.plot_xy_comparison(
                    xy_data, physics_xy_full, hybrid_xy_full, start_idx, hist_len,
                    save_path=os.path.join(method_save_dir, 'xy_comparison.png'),
                    title_suffix=f' ({trip_id} - {method_name})'
                )
                self.visualizer.visualize_model_comparison(
                    physics_preds_full, hybrid_preds_full, state_data, start_idx, hist_len,
                    save_path=os.path.join(method_save_dir, 'state_comparison.png')
                )
                self.visualizer.plot_error_curves(
                    cumulative_errors, start_idx, hist_len,
                    save_path=os.path.join(method_save_dir, 'error_curves.png')
                )
            return {
                'predictions': predictions, 'full_predictions': hybrid_preds_full,
                'targets_compare': targets, 'deltas': deltas,
                'cumulative_errors': cumulative_errors, 'xy_predict': xy_predict,
                'full_xy_predict': hybrid_xy_full
            }

        res = process_method("1_Interactive", preds, deltas, targets, cum_errors, xy_preds)
        res_add = process_method("2_Additive", preds_add, deltas_add, targets_add, cum_errors_add, xy_preds_add)
        
        gt_state = state_data[compare_start:] if compare_start < len(state_data) else np.array([])
        gt_xy = xy_data[compare_start:] if compare_start < len(xy_data) else np.array([])
        phy_state = physics_preds_full[compare_start:] if compare_start < len(physics_preds_full) else np.array([])
        phy_xy = physics_xy_full[compare_start:] if compare_start < len(physics_xy_full) else np.array([])

        if len(gt_state) > 0 and len(phy_state) > 0:
            phy_state_err = np.abs(phy_state - gt_state)
            physics_errors = {
                'rmse': np.sqrt(np.mean(phy_state_err**2, axis=0)),
                'mae': np.mean(phy_state_err, axis=0)
            }
        else:
            physics_errors = {'rmse': np.zeros(7), 'mae': np.zeros(7)}

        if len(gt_xy) > 0 and len(phy_xy) > 0:
            phy_dist_err = np.sqrt((phy_xy[:, 0] - gt_xy[:, 0])**2 + (phy_xy[:, 1] - gt_xy[:, 1])**2)
            phy_xy_errors = {
                'x': {'rmse': np.sqrt(np.mean((phy_xy[:, 0] - gt_xy[:, 0])**2)), 
                      'mae': np.mean(np.abs(phy_xy[:, 0] - gt_xy[:, 0])), 
                      'max': np.max(np.abs(phy_xy[:, 0] - gt_xy[:, 0]))},
                'y': {'rmse': np.sqrt(np.mean((phy_xy[:, 1] - gt_xy[:, 1])**2)), 
                      'mae': np.mean(np.abs(phy_xy[:, 1] - gt_xy[:, 1])), 
                      'max': np.max(np.abs(phy_xy[:, 1] - gt_xy[:, 1]))},
                'distance': {'rmse': np.sqrt(np.mean(phy_dist_err**2)), 
                             'mae': np.mean(phy_dist_err), 
                             'max': np.max(phy_dist_err)}
            }
        else:
            phy_xy_errors = {
                'x': {'rmse': 0, 'mae': 0, 'max': 0},
                'y': {'rmse': 0, 'mae': 0, 'max': 0},
                'distance': {'rmse': 0, 'mae': 0, 'max': 0}
            }

        def evaluate_and_print(method_name, res_dict):
            hyb_state = res_dict['full_predictions'][compare_start:] if compare_start < len(res_dict['full_predictions']) else np.array([])
            
            if len(gt_state) > 0 and len(hyb_state) > 0:
                hyb_state_err = np.abs(hyb_state - gt_state)
                hybrid_errors = {
                    'rmse': np.sqrt(np.mean(hyb_state_err**2, axis=0)),
                    'mae': np.mean(hyb_state_err, axis=0)
                }
            else:
                hybrid_errors = {'rmse': np.zeros(7), 'mae': np.zeros(7)}
            
            hyb_xy = res_dict['full_xy_predict'][compare_start:] if compare_start < len(res_dict['full_xy_predict']) else np.array([])
            
            if len(gt_xy) > 0 and len(hyb_xy) > 0:
                hyb_dist_err = np.sqrt((hyb_xy[:, 0] - gt_xy[:, 0])**2 + (hyb_xy[:, 1] - gt_xy[:, 1])**2)
                hyb_xy_errors = {
                    'x': {'rmse': np.sqrt(np.mean((hyb_xy[:, 0] - gt_xy[:, 0])**2)), 
                          'mae': np.mean(np.abs(hyb_xy[:, 0] - gt_xy[:, 0])), 
                          'max': np.max(np.abs(hyb_xy[:, 0] - gt_xy[:, 0]))},
                    'y': {'rmse': np.sqrt(np.mean((hyb_xy[:, 1] - gt_xy[:, 1])**2)), 
                          'mae': np.mean(np.abs(hyb_xy[:, 1] - gt_xy[:, 1])), 
                          'max': np.max(np.abs(hyb_xy[:, 1] - gt_xy[:, 1]))},
                    'distance': {'rmse': np.sqrt(np.mean(hyb_dist_err**2)), 
                                 'mae': np.mean(hyb_dist_err), 
                                 'max': np.max(hyb_dist_err)}
                }
            else:
                hyb_xy_errors = {
                    'x': {'rmse': 0, 'mae': 0, 'max': 0},
                    'y': {'rmse': 0, 'mae': 0, 'max': 0},
                    'distance': {'rmse': 0, 'mae': 0, 'max': 0}
                }

            imp_x = (phy_xy_errors['x']['rmse'] - hyb_xy_errors['x']['rmse']) / phy_xy_errors['x']['rmse'] * 100 if phy_xy_errors['x']['rmse'] > 0 else 0
            imp_y = (phy_xy_errors['y']['rmse'] - hyb_xy_errors['y']['rmse']) / phy_xy_errors['y']['rmse'] * 100 if phy_xy_errors['y']['rmse'] > 0 else 0
            imp_dist = (phy_xy_errors['distance']['rmse'] - hyb_xy_errors['distance']['rmse']) / phy_xy_errors['distance']['rmse'] * 100 if phy_xy_errors['distance']['rmse'] > 0 else 0

            print(f"\n[{method_name}] XY位置误差对比结果:")
            print("-"*115)
            print(f"{'Model':<10} {'Axis':<6} {'RMSE':<12} {'MAE':<12} {'Max Error':<12} {'RMSE改进':<12}")
            print("-"*115)
            print(f"{'Physics':<10} {'x':<6} {phy_xy_errors['x']['rmse']:<12.6f} {phy_xy_errors['x']['mae']:<12.6f} {phy_xy_errors['x']['max']:<12.6f} {'--':<12}")
            print(f"{'Hybrid':<10} {'x':<6} {hyb_xy_errors['x']['rmse']:<12.6f} {hyb_xy_errors['x']['mae']:<12.6f} {hyb_xy_errors['x']['max']:<12.6f} {imp_x:<12.1f}%")
            print(f"{'Physics':<10} {'y':<6} {phy_xy_errors['y']['rmse']:<12.6f} {phy_xy_errors['y']['mae']:<12.6f} {phy_xy_errors['y']['max']:<12.6f} {'--':<12}")
            print(f"{'Hybrid':<10} {'y':<6} {hyb_xy_errors['y']['rmse']:<12.6f} {hyb_xy_errors['y']['mae']:<12.6f} {hyb_xy_errors['y']['max']:<12.6f} {imp_y:<12.1f}%")
            print(f"{'Physics':<10} {'dist':<6} {phy_xy_errors['distance']['rmse']:<12.6f} {phy_xy_errors['distance']['mae']:<12.6f} {phy_xy_errors['distance']['max']:<12.6f} {'--':<12}")
            print(f"{'Hybrid':<10} {'dist':<6} {hyb_xy_errors['distance']['rmse']:<12.6f} {hyb_xy_errors['distance']['mae']:<12.6f} {hyb_xy_errors['distance']['max']:<12.6f} {imp_dist:<12.1f}%")
            print("-"*115)

            print(f"\n[{method_name}] 状态误差对比结果 (从t={compare_start}开始):")
            print("-"*105)
            print(f"{'Feature':<10} {'Physics RMSE':<15} {'Hybrid RMSE':<15} {'Improvement':<15} {'Physics MAE':<15} {'Hybrid MAE':<15} {'Improvement':<15}")
            print("-"*105)
            improvement = {}
            n_features = min(len(self.feature_name), len(physics_errors['rmse']))
            for i in range(n_features):
                name = self.feature_name[i]
                imp_rmse = (physics_errors['rmse'][i] - hybrid_errors['rmse'][i]) / physics_errors['rmse'][i] * 100 if physics_errors['rmse'][i] > 0 else 0
                imp_mae = (physics_errors['mae'][i] - hybrid_errors['mae'][i]) / physics_errors['mae'][i] * 100 if physics_errors['mae'][i] > 0 else 0
                improvement[name] = {'rmse_improvement': imp_rmse, 'mae_improvement': imp_mae}
                
                print(f"{name:<10} "
                      f"{physics_errors['rmse'][i]:<15.6f} "
                      f"{hybrid_errors['rmse'][i]:<15.6f} "
                      f"{imp_rmse:<15.1f}% "
                      f"{physics_errors['mae'][i]:<15.6f} "
                      f"{hybrid_errors['mae'][i]:<15.6f} "
                      f"{imp_mae:<15.1f}%")
            
            if n_features > 0:
                avg_rmse_imp = np.mean([improvement[name]['rmse_improvement'] for name in self.feature_name[:n_features]])
                avg_mae_imp = np.mean([improvement[name]['mae_improvement'] for name in self.feature_name[:n_features]])
                print(f"平均RMSE改进: {avg_rmse_imp:.1f}% | 平均MAE改进: {avg_mae_imp:.1f}%")
            print("-"*105)

        evaluate_and_print("1_Interactive", res)
        evaluate_and_print("2_Additive", res_add)

        if save_path is not None:
            trip_save_dir = os.path.join(save_path, trip_id)
            self.visualizer.visualize_data_comparison(state_data, base_data, control_data, save_path=trip_save_dir)
            
        return {
            'trip_id': trip_id, 'targets': state_data, 'gt_xy': xy_data,  # Ground Truth XY
            'physics_preds_full': physics_preds_full,
            'control_data': control_data, 'physics_xy_full': physics_xy_full,
            'start_idx': start_idx, 'hist_len': hist_len,
            'interactive': res, 'additive': res_add
        }


    def generate_inference_report(self, all_results, save_path, total_time, start_idx, hist_len):
        """生成推理效果统计报告"""
        os.makedirs(save_path, exist_ok=True)
        
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("                     DyTR 推理效果统计报告")
        report_lines.append("=" * 100)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"总处理 Trip 数: {len(all_results)}")
        report_lines.append(f"总推理时间: {total_time:.2f} 秒")
        report_lines.append(f"平均每 Trip 推理时间: {total_time/len(all_results):.2f} 秒")
        report_lines.append(f"预测起始索引: {start_idx}")
        report_lines.append(f"历史序列长度: {hist_len}")
        report_lines.append("=" * 100)
        
        # 统计所有 trip 的总体误差
        all_phy_state_rmse, all_hyb_state_rmse = [], []
        all_phy_xy_rmse, all_hyb_xy_rmse = [], []
        
        for res in all_results:
            gt_state = res['targets']
            phy_state = res['physics_preds_full']
            compare_start = start_idx + hist_len
            
            if compare_start < len(gt_state) and compare_start < len(phy_state):
                phy_state_err = gt_state[compare_start:] - phy_state[compare_start:]
                all_phy_state_rmse.append(np.sqrt(np.mean(phy_state_err**2, axis=0)))
            
            for method_key in ['interactive', 'additive']:
                hyb_preds = res[method_key]['full_predictions']
                if compare_start < len(gt_state) and compare_start < len(hyb_preds):
                    hyb_state_err = gt_state[compare_start:] - hyb_preds[compare_start:]
                    if method_key == 'interactive':
                        all_hyb_state_rmse.append(np.sqrt(np.mean(hyb_state_err**2, axis=0)))
        
        # 每个 trip 的详细统计
        trip_stats = []
        for res in all_results:
            gt_state = res['targets']
            phy_state = res['physics_preds_full']
            gt_xy = res['gt_xy']
            phy_xy = res['physics_xy_full']
            compare_start = start_idx + hist_len
            
            trip_stat = {
                'trip_id': res['trip_id'],
                'n_samples': res.get('n_samples', len(gt_state)),
                'inference_time': res.get('inference_time', 0),
                'pred_samples': len(gt_state) - compare_start if compare_start < len(gt_state) else 0
            }
            
            # 物理模型状态误差
            if compare_start < len(gt_state) and compare_start < len(phy_state):
                phy_state_err = gt_state[compare_start:] - phy_state[compare_start:]
                trip_stat['phy_state_rmse'] = np.sqrt(np.mean(phy_state_err**2))
                trip_stat['phy_state_mse'] = np.mean(np.sum(phy_state_err**2, axis=1))
            else:
                trip_stat['phy_state_rmse'] = 0
                trip_stat['phy_state_mse'] = 0
            
            # XY 轨迹误差
            if compare_start < len(gt_xy) and compare_start < len(phy_xy):
                phy_xy_err = gt_xy[compare_start:] - phy_xy[compare_start:]
                trip_stat['phy_xy_rmse'] = np.sqrt(np.mean(np.sum(phy_xy_err**2, axis=1)))
                trip_stat['phy_xy_mse'] = np.mean(np.sum(phy_xy_err**2, axis=1))
            else:
                trip_stat['phy_xy_rmse'] = 0
                trip_stat['phy_xy_mse'] = 0
            
            # 混合模型 (Interactive) 误差
            if 'interactive' in res:
                hyb_preds = res['interactive']['full_predictions']
                if compare_start < len(gt_state) and compare_start < len(hyb_preds):
                    hyb_state_err = gt_state[compare_start:] - hyb_preds[compare_start:]
                    trip_stat['hyb_state_rmse'] = np.sqrt(np.mean(hyb_state_err**2))
                    trip_stat['hyb_state_mse'] = np.mean(np.sum(hyb_state_err**2, axis=1))
                    trip_stat['state_improvement'] = ((trip_stat['phy_state_rmse'] - trip_stat['hyb_state_rmse']) / trip_stat['phy_state_rmse'] * 100) if trip_stat['phy_state_rmse'] > 0 else 0
                else:
                    trip_stat['hyb_state_rmse'] = 0
                    trip_stat['hyb_state_mse'] = 0
                    trip_stat['state_improvement'] = 0
                
                hyb_xy = res['interactive']['full_xy_predict']
                if compare_start < len(gt_xy) and compare_start < len(hyb_xy):
                    hyb_xy_err = gt_xy[compare_start:] - hyb_xy[compare_start:]
                    trip_stat['hyb_xy_rmse'] = np.sqrt(np.mean(np.sum(hyb_xy_err**2, axis=1)))
                    trip_stat['hyb_xy_mse'] = np.mean(np.sum(hyb_xy_err**2, axis=1))
                    trip_stat['xy_improvement'] = ((trip_stat['phy_xy_rmse'] - trip_stat['hyb_xy_rmse']) / trip_stat['phy_xy_rmse'] * 100) if trip_stat['phy_xy_rmse'] > 0 else 0
                else:
                    trip_stat['hyb_xy_rmse'] = 0
                    trip_stat['hyb_xy_mse'] = 0
                    trip_stat['xy_improvement'] = 0
            else:
                trip_stat['hyb_state_rmse'] = 0
                trip_stat['hyb_state_mse'] = 0
                trip_stat['state_improvement'] = 0
                trip_stat['hyb_xy_rmse'] = 0
                trip_stat['hyb_xy_mse'] = 0
                trip_stat['xy_improvement'] = 0
            
            trip_stats.append(trip_stat)
        
        # 生成报告表格
        report_lines.append("\n" + "-" * 100)
        report_lines.append("                              各 Trip 推理效果详细统计")
        report_lines.append("-" * 100)
        report_lines.append(f"{'Trip ID':<20} {'样本数':<10} {'预测样本':<10} {'推理时间(s)':<12} {'状态RMSE':<12} {'混合RMSE':<12} {'改进%':<10}")
        report_lines.append("-" * 100)
        
        for stat in trip_stats:
            report_lines.append(f"{stat['trip_id']:<20} {stat['n_samples']:<10} {stat['pred_samples']:<10} "
                              f"{stat['inference_time']:<12.2f} {stat['phy_state_rmse']:<12.4f} "
                              f"{stat['hyb_state_rmse']:<12.4f} {stat['state_improvement']:<10.1f}")
        
        report_lines.append("-" * 100)
        
        # 计算平均值
        avg_phy_state_rmse = np.mean([s['phy_state_rmse'] for s in trip_stats])
        avg_hyb_state_rmse = np.mean([s['hyb_state_rmse'] for s in trip_stats])
        avg_phy_xy_rmse = np.mean([s['phy_xy_rmse'] for s in trip_stats])
        avg_hyb_xy_rmse = np.mean([s['hyb_xy_rmse'] for s in trip_stats])
        avg_inference_time = np.mean([s['inference_time'] for s in trip_stats])
        avg_state_improvement = np.mean([s['state_improvement'] for s in trip_stats])
        avg_xy_improvement = np.mean([s['xy_improvement'] for s in trip_stats])
        
        report_lines.append(f"{'平均值':<20} {'-':<10} {'-':<10} {avg_inference_time:<12.2f} {avg_phy_state_rmse:<12.4f} {avg_hyb_state_rmse:<12.4f} {avg_state_improvement:<10.1f}")
        report_lines.append("-" * 100)
        
        # 总体改进统计
        report_lines.append("\n" + "=" * 100)
        report_lines.append("                              总体改进统计")
        report_lines.append("=" * 100)
        report_lines.append(f"物理模型平均状态 RMSE:  {avg_phy_state_rmse:.4f}")
        report_lines.append(f"混合模型平均状态 RMSE:  {avg_hyb_state_rmse:.4f}")
        report_lines.append(f"状态预测平均改进率:    {avg_state_improvement:.2f}%")
        report_lines.append("")
        report_lines.append(f"物理模型平均XY轨迹RMSE: {avg_phy_xy_rmse:.4f} m")
        report_lines.append(f"混合模型平均XY轨迹RMSE: {avg_hyb_xy_rmse:.4f} m")
        report_lines.append(f"轨迹预测平均改进率:     {avg_xy_improvement:.2f}%")
        report_lines.append("")
        report_lines.append(f"平均推理时间:           {avg_inference_time:.4f} 秒/Trip")
        report_lines.append(f"平均推理速度:           {np.mean([s['n_samples'] for s in trip_stats])/avg_inference_time:.1f} 样本/秒")
        report_lines.append("=" * 100)
        
        # 保存文本报告
        report_text = "\n".join(report_lines)
        report_file = os.path.join(save_path, "inference_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n推理报告已保存到: {report_file}")
        
        # 保存 JSON 报告（方便程序读取）
        json_report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_trips': len(all_results),
            'total_time': total_time,
            'avg_inference_time': avg_inference_time,
            'params': {
                'start_idx': start_idx,
                'hist_len': hist_len
            },
            'summary': {
                'physics_state_rmse': avg_phy_state_rmse,
                'hybrid_state_rmse': avg_hyb_state_rmse,
                'state_improvement': avg_state_improvement,
                'physics_xy_rmse': avg_phy_xy_rmse,
                'hybrid_xy_rmse': avg_hyb_xy_rmse,
                'xy_improvement': avg_xy_improvement,
                'avg_samples_per_second': np.mean([s['n_samples'] for s in trip_stats])/avg_inference_time
            },
            'trip_details': trip_stats
        }
        
        json_file = os.path.join(save_path, "inference_report.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, ensure_ascii=False, indent=2)
        print(f"JSON报告已保存到: {json_file}")
        
        # 保存 CSV 报告
        try:
            import pandas as pd
            df = pd.DataFrame(trip_stats)
            csv_file = os.path.join(save_path, "inference_report.csv")
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print(f"CSV报告已保存到: {csv_file}")
        except ImportError:
            print("提示: 安装 pandas 可以生成 CSV 格式报告 (pip install pandas)")
        
        # 打印报告
        print("\n" + report_text)

    def run_batch_inference(self, data_dir, cfg_value=2237.9, start_idx=0, hist_len=15, max_trips=None, save_path=None):
        pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
        if not pkl_files:
            print(f"在目录 {data_dir} 中未找到pkl文件")
            return []
        print(f"找到 {len(pkl_files)} 个pkl文件")
        if max_trips is not None:
            pkl_files = pkl_files[-max_trips:]
            
        all_results = []
        total_start_time = time.time()
        
        for i, pkl_file in enumerate(pkl_files):
            print(f"\n处理 {i+1}/{len(pkl_files)}: {os.path.basename(pkl_file)}")
            try:
                trip_start_time = time.time()
                result = self.run_single_trip_inference(pkl_file, cfg_value, start_idx, hist_len, save_path)
                trip_end_time = time.time()
                result['inference_time'] = trip_end_time - trip_start_time
                result['n_samples'] = len(result['targets'])
                all_results.append(result)
            except Exception as e:
                print(f"处理失败: {e}")
                import traceback
                traceback.print_exc()
        
        total_end_time = time.time()
        total_inference_time = total_end_time - total_start_time
        
        if all_results and save_path:
            self.generate_inference_report(all_results, save_path, total_inference_time, start_idx, hist_len)
                
        if all_results:
            def format_for_plotting(method_key):
                formatted_results = []
                for res in all_results:
                    formatted_results.append({
                        'trip_id': res['trip_id'],
                        'gt_xy': res['gt_xy'],  # Ground Truth XY
                        'physics_xy_full': res['physics_xy_full'],
                        'physics_preds_full': res['physics_preds_full'],
                        'targets': res['targets'],
                        'control_data': res['control_data'],
                        'hybrid_xy_full': res[method_key]['full_xy_predict'],
                        'hybrid_preds_full': res[method_key]['full_predictions'],
                        'xy_predict': res[method_key]['xy_predict'],
                        'cumulative_errors': res[method_key]['cumulative_errors'],
                        'deltas': res[method_key]['deltas'],
                    })
                return formatted_results

            print("\n生成 [交互迭代] 的批量汇总图...")
            inter_data = format_for_plotting('interactive')
            inter_save_dir = os.path.join(save_path, "Batch_Summary_Interactive") if save_path else None
            if inter_save_dir:
                os.makedirs(inter_save_dir, exist_ok=True)
            self.visualizer.visualize_combined_3d_scatter(inter_data, inter_save_dir)
            self.visualizer.visualize_error_comparison(inter_data, inter_save_dir)
            self.visualizer.visualize_error_vs_control_angle(inter_data, inter_save_dir)
            self.visualizer.visualize_xy_reconstruction_comparison(inter_data, inter_save_dir)
            self.visualizer.visualize_xy_comparison_fixed_gt(inter_data, inter_save_dir)

            print("\n生成 [分别迭代相加] 的批量汇总图...")
            add_data = format_for_plotting('additive')
            add_save_dir = os.path.join(save_path, "Batch_Summary_Additive") if save_path else None
            if add_save_dir:
                os.makedirs(add_save_dir, exist_ok=True)
            self.visualizer.visualize_combined_3d_scatter(add_data, add_save_dir)
            self.visualizer.visualize_error_comparison(add_data, add_save_dir)
            self.visualizer.visualize_error_vs_control_angle(add_data, add_save_dir)
            self.visualizer.visualize_xy_reconstruction_comparison(add_data, add_save_dir)
            self.visualizer.visualize_xy_comparison_fixed_gt(add_data, add_save_dir)

        return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DyTR 推理脚本')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='输出目录 (默认: ./inference_results_mul，可传入训练输出目录)')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='模型文件路径 (默认: 自动从 latest_model_config.json 读取)')
    parser.add_argument('--model_type', type=str, default=None, 
                        help='模型类型: lstm 或 mlp (默认: 自动从配置文件读取最新模型)')
    parser.add_argument('--data_dir', type=str, default='./data/left_turn_dataset', 
                        help='数据目录 (默认: ./data/left_turn_dataset)')
    parser.add_argument('--max_trips', type=int, default=2, 
                        help='处理的trip数量 (默认: 2)')
    parser.add_argument('--start_idx', type=int, default=20, 
                        help='预测起始索引 (默认: 20)')
    parser.add_argument('--hist_len', type=int, default=15, 
                        help='历史序列长度 (默认: 15)')
    args = parser.parse_args()
    
    # 自动读取模型路径配置
    config_path = 'latest_model_config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"已读取模型配置: {config_path}")
        print(f"可用模型:")
        for key, value in config.items():
            if key not in ['latest', 'timestamp'] and isinstance(value, str) and value.endswith('.pth'):
                print(f"  - {key}: {value}")
    else:
        config = {}
        print(f"警告: 配置文件 {config_path} 不存在，将使用默认模型路径")
    
    # 确定使用的模型
    if args.model_type:
        model_key = f'{args.model_type.upper()}_best_model'
        if model_key in config:
            args.model_path = config[model_key]
            print(f"使用指定的模型类型: {args.model_type.upper()}, 路径: {args.model_path}")
        else:
            print(f"警告: 配置中未找到 {model_key}，将使用默认路径")
            args.model_path = f'{args.model_type.lower()}_model.pth'
    elif args.model_path is None:
        # 自动使用最新训练的模型
        if 'latest' in config and config['latest']:
            model_key = f"{config['latest']}_best_model"
            if model_key in config:
                args.model_path = config[model_key]
                print(f"自动使用最新训练的模型: {config['latest'].upper()}, 路径: {args.model_path}")
            else:
                print(f"警告: 配置中未找到 {model_key}，将使用默认路径")
                args.model_path = 'left_turn_model.pth'
        else:
            args.model_path = 'left_turn_model.pth'
    
    # 确定输出目录
    if args.output_dir is None:
        if 'latest' in config and config['latest']:
            output_key = f"{config['latest']}_output_dir"
            if output_key in config:
                args.output_dir = os.path.join(config[output_key], 'inference_mul')
                print(f"自动设置输出目录: {args.output_dir}")
            else:
                args.output_dir = './inference_results_mul'
        else:
            args.output_dir = './inference_results_mul'
    
    print(f"\n推理配置:")
    print(f"  模型路径: {args.model_path}")
    print(f"  数据目录: {args.data_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  处理Trip数: {args.max_trips}")
    
    model_flag = 3
    if model_flag == 3:
        model_config = {
            'state_dim': 4, 'control_dim': 2, 
            'feature_dim': 64, 'hist_len': 15, 'pred_len': 1
        }

    feature_name = ["vlon", "vlat", "yaw", "vyaw", "acc_lon", "acc_lat", "acc_yaw"]
    control_name = ["acc", "angle"]

    data_dir = args.data_dir
    save_path = args.output_dir if args.output_dir else "./inference_results_mul"
    model_path = args.model_path
    scaler_path = None
    
    # 如果传入的是训练输出目录，在下面创建 inference 子目录
    if args.output_dir and 'outputs' in args.output_dir:
        save_path = os.path.join(args.output_dir, 'inference_mul')
    
    print(f"推理结果将保存到: {save_path}")

    infer_model = InferModel(
        model_config=model_config,
        model_flag=model_flag,
        model_path=model_path,
        scaler_path=scaler_path,
        dt=0.01,
        feature_name=feature_name,
        control_name=control_name,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    results = infer_model.run_batch_inference(
        data_dir=data_dir,
        cfg_value=2237.9,
        start_idx=args.start_idx, 
        hist_len=args.hist_len,
        max_trips=args.max_trips,
        save_path=save_path
    )