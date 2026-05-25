import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pickle
import joblib
from scipy import signal
import sys
sys.path.append('/home/luban/dytr/dytr_project/residual_model')
from informer_models_elements.dytr_model import DyTR
from other_model.lstm_dytr import DyTR_LSTM
from other_model.mlp_dytr import DyTR_MLP
from input_data_preprocessing.python.voy_offboard_control.models import dynamic_bicycle

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
        self.phy_model = dynamic_bicycle.LateralDynamicBicycle(parameters=params)
        net_model = self.load_trained_model(model_path, model_config, device)
        self.model = net_model.to(device)
        self.model.eval()

        self.visualizer = InferVisualizer(
            feature_name=feature_name, 
            control_name=control_name, 
            dt=dt
        )

        self.scalers = None
        if scaler_path is not None:
            self.scalers = joblib.load(scaler_path)
            print("Scaler 加载成功，包含的键:", self.scalers.keys())

    def read_real_and_dynamic_data(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        [model_states_real, model_states_base, control_input] = data[1]
        state_data = model_states_real[:, 2:]   # vlon, vlat, yaw, vyaw, acc_lon, acc_lat, acc_yaw
        base_data = model_states_base[:, 2:]     # 原始物理模型输出（7维）
        control_data = control_input[:, :]      # acc, angle
        xy_data = model_states_real[:, :2]      # x, y
        return np.array(state_data), np.array(control_data), np.array(base_data), np.array(xy_data)

    def load_trained_model(self, model_path, model_config, device='cpu'):
        device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=device)
        print(f"checkpoint: epoch={checkpoint.get('epoch', 'unknown')}, val_loss={checkpoint.get('best_val_loss', 'unknown')}")
        if self.model_flag == 1:
            model = DyTR(**model_config)
        elif self.model_flag == 2:
            model = DyTR_MLP(**model_config)
        elif self.model_flag == 3:
            model = DyTR_LSTM(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
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
        cfg_tensor = cfg_tensor.to(self.device) if cfg_tensor is not None else None

        with torch.no_grad():
            delta_norm, corrected_norm = self.model(
                s_hist=hist_state_tensor,         
                u_hist=hist_control_tensor,      
                s_next=None,                      
                c=cfg_tensor,                    
                base_state=base_state_tensor, 
                return_corrected=True
            )
        corrected_norm = corrected_norm.squeeze(0).squeeze(0).cpu().numpy()
        delta_norm = delta_norm.squeeze(0).cpu().numpy()

        if self.scalers is not None:
            corrected = self.scalers['state'].inverse_transform(corrected_norm.reshape(1, -1)).flatten()
            delta = self.scalers['diff'].inverse_transform(delta_norm.reshape(1, -1)).flatten()
            base_state = self.scalers['base'].inverse_transform(base_state_norm.reshape(1, -1)).flatten()
        else:
            corrected, delta, base_state = corrected_norm, delta_norm, base_state_norm

        # print(f"归一化空间验证: base_norm={base_state_norm.flatten()}")
        # print(f"归一化空间验证: delta_norm={delta_norm.flatten()}")
        # print(f"归一化空间验证: corrected_norm={corrected_norm.flatten()}")
        # print(f"相加结果: {base_state_norm.flatten() + delta_norm.flatten()}")
        return corrected, delta, base_state


    def predict_physics_single_step(self, current_states, current_control, xy_data_t):
        _SEQUENCY_SIZE = 1
        state_input = self.phy_model.State(np.zeros([len(self.phy_model.States), _SEQUENCY_SIZE]))
        control_input = self.phy_model.ControlInput(np.zeros([len(self.phy_model.ControlInputs), _SEQUENCY_SIZE]))

        control_T = current_control[-1, :].T.reshape(-1, 1)
        current_states_T = current_states[-1, :].T.reshape(-1, 1)
        xy_data_T = xy_data_t.T

        state_input[0, :] = xy_data_T[0, :]          
        state_input[1, :] = xy_data_T[1, :]          
        state_input[2, :] = current_states_T[0, :]   
        state_input[3, :] = current_states_T[1, :]   
        state_input[4, :] = current_states_T[2, :]   
        state_input[5, :] = current_states_T[3, :]   
        for i in range(2):
            control_input[i, :] = control_T[i, :]

        new_states, _ = self.phy_model.simulate(state_input[:, 0], control_input, n=2, dt=self.dt)

        acc_lon = (new_states[2, 1:2] - state_input[2, :]) / self.dt
        acc_lat = (new_states[3, 1:2] - state_input[3, :]) / self.dt
        acc_yaw = (new_states[5, 1:2] - state_input[5, :]) / self.dt

        xy_data_t = new_states[:2, 1:2].T
        phy_pred = np.concatenate([
            new_states[2:, 1:2].T,   
            acc_lon.reshape(1, -1), acc_lat.reshape(1, -1), acc_yaw.reshape(1, -1)
        ], axis=1)
        return phy_pred, xy_data_t

    def run_physics_only_full(self, state_data, control_data, xy_data):
        n_samples = len(state_data)
        physics_states = []
        physics_xy = []
        current_hist_state = state_data[0:1, :]
        current_hist_control = control_data[0:1, :]
        xy_data_t = xy_data[0:1, :]

        physics_states.append(current_hist_state[0])
        physics_xy.append(xy_data_t[0])

        for t in range(1, n_samples):
            physics_pred, xy_data_phy = self.predict_physics_single_step(current_hist_state, current_hist_control, xy_data_t)
            if t < n_samples:
                current_hist_control = np.vstack([current_hist_control[1:, :], control_data[t:t+1, :]])
                xy_data_t = xy_data_phy
                current_hist_state = np.vstack([current_hist_state[1:, :], physics_pred])
            physics_states.append(physics_pred[0])
            physics_xy.append(xy_data_phy[0])

        physics_states = np.array(physics_states)
        tmp_norm_angles = self.normalize_angle_sequence(physics_states[:, 2])
        physics_states[:, 2] = tmp_norm_angles.flatten()
        return physics_states, np.array(physics_xy)

    def autoregressive_predict(self, state_data, control_data, base_data, xy_data,
                               start_idx, hist_len=15, cfg_data=None):
        n_samples = len(state_data)
        predictions, xy_predict, deltas, targets, cumulative_errors = [], [], [], [], []

        physics_states, physics_xy = self.run_physics_only_full(
            state_data[:start_idx+hist_len], control_data[:start_idx+hist_len], xy_data[:start_idx+hist_len]
        )

        current_hist_state = physics_states[start_idx:start_idx+hist_len, :]
        current_hist_control = control_data[start_idx:start_idx+hist_len, :]
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
                # delta[1] = correct[1] - base[1]
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
                current_hist_control = np.vstack([current_hist_control[1:, :], control_data[t:t+1, :]])
                xy_data_t = np.array([[data_x, data_y]])
                current_hist_state = np.vstack([current_hist_state[1:, :], physics_pred])
            prev_yaw = physics_pred[0, 2]
            xy_predict.append([data_x, data_y])
            predictions.append(physics_pred[0])
            deltas.append(delta_data)
            targets.append(state_data[t])
            cumulative_errors.append(physics_pred[0] - state_data[t])
            
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
            hist_control = control_data[t-hist_len:t, :]

            # hist_state = state_data[t-hist_len:t, :]
            # # 注意：predict_net_single_step 里的 base_state 也用真实值
            # current_gt_state = state_data[t:t+1, :] 
            # hist_control = control_data[t-hist_len:t, :]
            
            
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
            targets.append(state_data[t])
            cumulative_errors.append(mixed_pred[0] - state_data[t])
            
        print("\n混合模型预测完成!")
        predictions = np.array(predictions)
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
                hybrid_xy_full[compare_start:compare_start+len(xy_predict)] = xy_predict
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
        
        # ==========================================
        # 统一计算基准误差 并打印结果
        # ==========================================
        gt_state = state_data[compare_start:]
        gt_xy = xy_data[compare_start:]
        phy_state = physics_preds_full[compare_start:]
        phy_xy = physics_xy_full[compare_start:]

        # 1. 计算物理模型状态误差
        phy_state_err = np.abs(phy_state - gt_state)
        physics_errors = {
            'rmse': np.sqrt(np.mean(phy_state_err**2, axis=0)),
            'mae': np.mean(phy_state_err, axis=0)
        }

        # 2. 计算物理模型 XY 误差
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

        def evaluate_and_print(method_name, res_dict):
            hyb_state = res_dict['full_predictions'][compare_start:]
            hyb_state_err = np.abs(hyb_state - gt_state)
            hybrid_errors = {
                'rmse': np.sqrt(np.mean(hyb_state_err**2, axis=0)),
                'mae': np.mean(hyb_state_err, axis=0)
            }
            
            hyb_xy = res_dict['full_xy_predict'][compare_start:]
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

            # --- 4. 打印 XY 误差 ---
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

            # --- 5. 打印状态误差对比结果 ---
            print(f"\n[{method_name}] 状态误差对比结果 (从t={compare_start}开始):")
            print("-"*105)
            print(f"{'Feature':<10} {'Physics RMSE':<15} {'Hybrid RMSE':<15} {'Improvement':<15} {'Physics MAE':<15} {'Hybrid MAE':<15} {'Improvement':<15}")
            print("-"*105)
            improvement = {}
            for i, name in enumerate(self.feature_name):
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
            
            avg_rmse_imp = np.mean([improvement[name]['rmse_improvement'] for name in self.feature_name])
            avg_mae_imp = np.mean([improvement[name]['mae_improvement'] for name in self.feature_name])
            print(f"平均RMSE改进: {avg_rmse_imp:.1f}% | 平均MAE改进: {avg_mae_imp:.1f}%")
            print("-"*105)

        # 分别对两种模式进行评估和打印
        evaluate_and_print("1_Interactive", res)
        evaluate_and_print("2_Additive", res_add)

        if save_path is not None:
            trip_save_dir = os.path.join(save_path, trip_id)
            self.visualizer.visualize_data_comparison(state_data, base_data, control_data, save_path=trip_save_dir)
            
        return {
            'trip_id': trip_id, 'targets': state_data, 'physics_preds_full': physics_preds_full,
            'control_data': control_data, 'physics_xy_full': physics_xy_full,
            'start_idx': start_idx, 'hist_len': hist_len,
            'interactive': res, 'additive': res_add
        }


    def run_batch_inference(self, data_dir, cfg_value=2237.9, start_idx=0, hist_len=15, max_trips=None, save_path=None):
        pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
        if not pkl_files:
            print(f"在目录 {data_dir} 中未找到pkl文件")
            return []
        print(f"找到 {len(pkl_files)} 个pkl文件")
        if max_trips is not None:
            pkl_files = pkl_files[-max_trips:]
            
        all_results = []
        for i, pkl_file in enumerate(pkl_files):
            print(f"\n处理 {i+1}/{len(pkl_files)}: {os.path.basename(pkl_file)}")
            try:
                result = self.run_single_trip_inference(pkl_file, cfg_value, start_idx, hist_len, save_path)
                all_results.append(result)
            except Exception as e:
                print(f"处理失败: {e}")
                
        if all_results:
            def format_for_plotting(method_key):
                formatted_results = []
                for res in all_results:
                    formatted_results.append({
                        'trip_id': res['trip_id'],
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
    # 配置参数
    model_flag = 3
    if model_flag == 3:
        model_config = {
            'state_dim': 4, 'control_dim': 2, 'config_dim': 1,
            'feature_dim': 4, 'T': 15, 'pred_len': 1,
            'layer_nums': 1, 'pos_kind': "learn", 'dropout_rate': 0.5
        }

    feature_name = ["vlon", "vlat", "yaw", "vyaw", "acc_lon", "acc_lat", "acc_yaw"]
    control_name = ["acc", "angle"]

    path = "/home/luban/dytr/giftdata"
    data_dir = "/home/luban/dytr/giftdata/Data/aqData/all_turn_right"
    save_path = f"{path}/dytr_project/residual_model/aqresults/model_3_norm_True_num_epochs200_batch_size256_20260410_1609"
    model_path = f"{save_path}/checkpoints/best_dytr_model.pth"
    scaler_path = f"{save_path}/checkpoints/scalers.pkl"

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
        start_idx=20, 
        hist_len=15,
        max_trips=2,
        save_path=save_path
    )