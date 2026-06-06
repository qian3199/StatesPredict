import matplotlib.pyplot as plt
import numpy as np
import os

class InferVisualizer:
    def __init__(self, feature_name=None, control_name=None, dt=0.01):
        self.feature_name = feature_name if feature_name else ["vlon", "vlat", "yaw", "vyaw", "acc_lon", "acc_lat", "acc_yaw"]
        self.control_name = control_name if control_name else ["acc", "angle"]
        self.dt = dt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_xy_comparison(self, gt_xy, physics_xy, hybrid_xy, start_idx, hist_len, 
                          save_path='xy_comparison.png', title_suffix=''):
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        compare_start = start_idx + hist_len
        
        # 三条轨迹，线条加粗到 4-5
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], label='Real (Ground Truth)', 
                color='#2ECC71', linewidth=5, linestyle='-', alpha=0.95)
        ax.plot(physics_xy[:, 0], physics_xy[:, 1], label='Physics Model', 
                color='#3498DB', linewidth=4, linestyle='--', alpha=0.85)
        ax.plot(hybrid_xy[:, 0], hybrid_xy[:, 1], label='Hybrid Prediction', 
                color='#E74C3C', linewidth=4, linestyle='-.', alpha=0.85)
        
        # 起点和终点标记加大
        ax.scatter(gt_xy[0, 0], gt_xy[0, 1], color='#F39C12', marker='*', s=300, zorder=10, label='Start')
        ax.scatter(gt_xy[-1, 0], gt_xy[-1, 1], color='#9B59B6', marker='s', s=180, zorder=10, label='End')
        
        # 预测起点标记
        if compare_start < len(gt_xy):
            ax.scatter(gt_xy[compare_start, 0], gt_xy[compare_start, 1], 
                       color='#34495E', marker='o', s=150, zorder=10, label='Prediction Start')
        
        # 计算误差
        phy_dist = np.sqrt((gt_xy[:, 0] - physics_xy[:, 0])**2 + (gt_xy[:, 1] - physics_xy[:, 1])**2)
        hybrid_dist = np.sqrt((gt_xy[:, 0] - hybrid_xy[:, 0])**2 + (gt_xy[:, 1] - hybrid_xy[:, 1])**2)
        
        rmse_phy = np.sqrt(np.mean(phy_dist[compare_start:]**2)) if compare_start < len(phy_dist) else 0
        rmse_hybrid = np.sqrt(np.mean(hybrid_dist[compare_start:]**2)) if compare_start < len(hybrid_dist) else 0
        
        ax.set_xlabel('X (m)', fontsize=14)
        ax.set_ylabel('Y (m)', fontsize=14)
        ax.set_title(f'XY Trajectory Comparison{title_suffix}\nPhysics RMSE: {rmse_phy:.4f}m, Hybrid RMSE: {rmse_hybrid:.4f}m', fontsize=16)
        ax.legend(fontsize=14, loc='best')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axis('equal')
        
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    def visualize_model_comparison(self, physics_preds, hybrid_preds, state_data, 
                                   start_idx=0, hist_len=15, save_path='state_comparison.png'):
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        compare_start = start_idx + hist_len
        n_states = min(4, state_data.shape[1])
        
        fig, axes = plt.subplots(n_states, 1, figsize=(16, 12))
        if n_states == 1:
            axes = [axes]
        
        time_steps = np.arange(len(state_data))
        
        for i in range(n_states):
            ax = axes[i]
            # 三条线都加粗
            ax.plot(time_steps, state_data[:, i], label='Real (Ground Truth)', 
                    color='#2ECC71', linewidth=3, alpha=0.95)
            ax.plot(time_steps, physics_preds[:, i], label='Physics Model', 
                    color='#3498DB', linewidth=2.5, linestyle='--', alpha=0.85)
            ax.plot(time_steps, hybrid_preds[:, i], label='Hybrid Prediction', 
                    color='#E74C3C', linewidth=2.5, linestyle='-.', alpha=0.85)
            
            ax.axvline(x=compare_start, color='black', linestyle='--', alpha=0.5, linewidth=2, label='Prediction Start')
            
            rmse_base = np.sqrt(np.mean((physics_preds[compare_start:, i] - state_data[compare_start:, i])**2))
            rmse_hybrid = np.sqrt(np.mean((hybrid_preds[compare_start:, i] - state_data[compare_start:, i])**2))
            
            ax.set_title(f'{self.feature_name[i]} (Physics RMSE: {rmse_base:.4f}, Hybrid RMSE: {rmse_hybrid:.4f})', fontsize=14)
            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel(self.feature_name[i], fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    def plot_error_curves(self, cumulative_errors, start_idx=0, hist_len=15, 
                          save_path='error_curves.png'):
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        compare_start = start_idx + hist_len
        n_features = min(4, cumulative_errors.shape[1])
        
        fig, axes = plt.subplots(n_features, 1, figsize=(14, 10))
        if n_features == 1:
            axes = [axes]
        
        time_steps = np.arange(len(cumulative_errors))
        
        for i in range(n_features):
            ax = axes[i]
            ax.plot(time_steps, cumulative_errors[:, i], label=f'{self.feature_name[i]} Error',
                    color='#E74C3C', linewidth=1.5)
            
            rmse = np.sqrt(np.mean(cumulative_errors[:, i]**2))
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.axhline(y=rmse, color='#F39C12', linestyle=':', alpha=0.7, label=f'RMSE: {rmse:.6f}')
            ax.axhline(y=-rmse, color='#F39C12', linestyle=':', alpha=0.7)
            
            ax.set_title(f'{self.feature_name[i]} Cumulative Error')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Error')
            ax.legend(fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_data_comparison(self, state_data, base_data, control_data, save_path='./'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        time_steps = np.arange(len(control_data))
        
        axes[0].plot(time_steps, control_data[:, 0], color='#3498DB', linewidth=2)
        axes[0].set_title('Control: Acceleration')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Acceleration')
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        axes[1].plot(time_steps, control_data[:, 1], color='#E74C3C', linewidth=2)
        axes[1].set_title('Control: Steering Angle')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Angle (rad)')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'control_inputs.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_combined_3d_scatter(self, results, save_path=None):
        if save_path is None:
            save_path = './'
        os.makedirs(save_path, exist_ok=True)
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for res in results:
            # 三条轨迹：Real、Physics、Hybrid
            gt_xy = res.get('gt_xy', res['physics_xy_full'])  # Ground Truth (Real)
            phy_xy = res['physics_xy_full']  # Physics Model
            hyb_xy = res['hybrid_xy_full']   # Hybrid Prediction
            
            time_steps = np.arange(len(gt_xy))
            
            ax.plot(gt_xy[:, 0], gt_xy[:, 1], time_steps, 
                    color='#2ECC71', linewidth=3, alpha=0.9, label='Real')
            ax.plot(phy_xy[:, 0], phy_xy[:, 1], time_steps, 
                    color='#3498DB', linewidth=2.5, alpha=0.7, label='Physics')
            ax.plot(hyb_xy[:, 0], hyb_xy[:, 1], time_steps, 
                    color='#E74C3C', linewidth=2.5, alpha=0.7, label='Hybrid')
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Time Step', fontsize=12)
        ax.set_title('3D Trajectory Comparison (Real vs Physics vs Hybrid)', fontsize=14)
        ax.legend(['Real', 'Physics', 'Hybrid'], fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, '3d_trajectory.png'), dpi=200, bbox_inches='tight')
        plt.close()
    
    def visualize_error_comparison(self, results, save_path=None):
        if save_path is None:
            save_path = './'
        os.makedirs(save_path, exist_ok=True)
        
        all_phy_errors = []
        all_hyb_errors = []
        
        for res in results:
            gt = res['targets']
            phy = res['physics_preds_full']
            hyb = res['hybrid_preds_full']
            
            phy_error = np.sqrt(np.mean((phy - gt)**2, axis=1))
            hyb_error = np.sqrt(np.mean((hyb - gt)**2, axis=1))
            
            all_phy_errors.append(phy_error)
            all_hyb_errors.append(hyb_error)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, (phy_err, hyb_err) in enumerate(zip(all_phy_errors, all_hyb_errors)):
            ax.plot(phy_err, color='#3498DB', linewidth=1, alpha=0.3)
            ax.plot(hyb_err, color='#E74C3C', linewidth=1, alpha=0.3)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('RMSE')
        ax.set_title('Error Comparison Across Trips')
        ax.legend(['Physics', 'Hybrid'])
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'error_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_error_vs_control_angle(self, results, save_path=None):
        if save_path is None:
            save_path = './'
        os.makedirs(save_path, exist_ok=True)
        
        all_angles = []
        all_phy_errors = []
        all_hyb_errors = []
        
        for res in results:
            angles = np.abs(res['control_data'][:, 1])
            gt = res['targets']
            phy = res['physics_preds_full']
            hyb = res['hybrid_preds_full']
            
            phy_error = np.sqrt(np.mean((phy - gt)**2, axis=1))
            hyb_error = np.sqrt(np.mean((hyb - gt)**2, axis=1))
            
            all_angles.extend(angles)
            all_phy_errors.extend(phy_error)
            all_hyb_errors.extend(hyb_error)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(all_angles, all_phy_errors, color='#3498DB', alpha=0.5, label='Physics Error')
        ax.scatter(all_angles, all_hyb_errors, color='#E74C3C', alpha=0.5, label='Hybrid Error')
        
        ax.set_xlabel('Steering Angle (abs)')
        ax.set_ylabel('RMSE')
        ax.set_title('Error vs Steering Angle')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'error_vs_angle.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_xy_reconstruction_comparison(self, results, save_path=None):
        if save_path is None:
            save_path = './'
        os.makedirs(save_path, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        for res in results:
            # 三条轨迹：Real、Physics、Hybrid
            gt_xy = res.get('gt_xy', res['physics_xy_full'])  # Ground Truth (Real)
            phy_xy = res['physics_xy_full']  # Physics Model
            hyb_xy = res['hybrid_xy_full']   # Hybrid Prediction
            
            ax.plot(gt_xy[:, 0], gt_xy[:, 1], color='#2ECC71', linewidth=3, alpha=0.8, label='Real')
            ax.plot(phy_xy[:, 0], phy_xy[:, 1], color='#3498DB', linewidth=2.5, linestyle='--', alpha=0.7, label='Physics')
            ax.plot(hyb_xy[:, 0], hyb_xy[:, 1], color='#E74C3C', linewidth=2.5, linestyle='-.', alpha=0.7, label='Hybrid')
        
        ax.set_title('XY Trajectory Reconstruction (Real vs Physics vs Hybrid)', fontsize=14)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.legend(['Real', 'Physics', 'Hybrid'], fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'xy_reconstruction.png'), dpi=200, bbox_inches='tight')
        plt.close()
    
    def visualize_xy_comparison_fixed_gt(self, results, save_path=None):
        if save_path is None:
            save_path = './'
        os.makedirs(save_path, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        for res in results:
            # 三条轨迹：Real、Physics、Hybrid
            gt_xy = res.get('gt_xy', res['physics_xy_full'])  # Ground Truth (Real)
            phy_xy = res['physics_xy_full']  # Physics Model
            hyb_xy = res['hybrid_xy_full']   # Hybrid Prediction
            
            # 归一化到起点
            start_x, start_y = gt_xy[0, 0], gt_xy[0, 1]
            gt_normalized = gt_xy - np.array([start_x, start_y])
            phy_normalized = phy_xy - np.array([start_x, start_y])
            hyb_normalized = hyb_xy - np.array([start_x, start_y])
            
            ax.plot(gt_normalized[:, 0], gt_normalized[:, 1], color='#2ECC71', linewidth=3, alpha=0.8, label='Real')
            ax.plot(phy_normalized[:, 0], phy_normalized[:, 1], color='#3498DB', linewidth=2.5, linestyle='--', alpha=0.7, label='Physics')
            ax.plot(hyb_normalized[:, 0], hyb_normalized[:, 1], color='#E74C3C', linewidth=2.5, linestyle='-.', alpha=0.7, label='Hybrid')
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('XY Trajectory Comparison (Normalized to Start) - Real vs Physics vs Hybrid', fontsize=14)
        ax.legend(['Real', 'Physics', 'Hybrid'], fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'xy_fixed_gt.png'), dpi=200, bbox_inches='tight')
        plt.close()