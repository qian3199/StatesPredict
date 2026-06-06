import matplotlib.pyplot as plt
import numpy as np
import os


class VisualUtils:
    def __init__(self, state_names=None, control_names=None, save_dir='./plots', dt=0.01):
        self.state_names = state_names if state_names else ['vlon', 'vlat', 'yaw', 'omega']
        self.control_names = control_names if control_names else ['acc', 'angle']
        self.save_dir = save_dir
        self.dt = dt
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def _get_save_path(self, save_path):
        if os.path.isabs(save_path):
            return save_path
        return os.path.join(self.save_dir, save_path)
    
    def plot_loss_curve(self, train_losses, val_losses=None, save_path='loss_curve.png'):
        save_path = self._get_save_path(save_path)
        
        epochs = np.arange(1, len(train_losses) + 1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, train_losses, label='Train Loss', color='#E74C3C', linewidth=2)
        
        if val_losses is not None and len(val_losses) > 0:
            ax.plot(epochs, val_losses, label='Val Loss', color='#3498DB', linewidth=2)
            
            min_val_epoch = np.argmin(val_losses) + 1
            min_val_loss = val_losses[min_val_epoch - 1]
            ax.scatter(min_val_epoch, min_val_loss, color='#2ECC71', s=100, marker='*', zorder=5)
            ax.annotate(f'Min: {min_val_loss:.4f}', 
                        xy=(min_val_epoch, min_val_loss),
                        xytext=(min_val_epoch + 2, min_val_loss + 0.05),
                        arrowprops=dict(arrowstyle='->', color='#2ECC71'))
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss Curve', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_state_comparison(self, pred_states, base_states, gt_states, save_path='state_comparison.png'):
        save_path = self._get_save_path(save_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        time_steps = np.arange(pred_states.shape[0])
        
        for i in range(4):
            ax = axes[i]
            
            ax.plot(time_steps, gt_states[:, i], label='Ground Truth', 
                    color='#2ECC71', linewidth=2, linestyle='-')
            ax.plot(time_steps, base_states[:, i], label='Physics Baseline', 
                    color='#3498DB', linewidth=2, linestyle='--')
            ax.plot(time_steps, pred_states[:, i], label='Hybrid Prediction', 
                    color='#E74C3C', linewidth=2, linestyle='-')
            
            rmse_base = np.sqrt(np.mean((base_states[:, i] - gt_states[:, i])**2))
            rmse_pred = np.sqrt(np.mean((pred_states[:, i] - gt_states[:, i])**2))
            
            ax.set_title(f'{self.state_names[i]} (Base RMSE: {rmse_base:.8f}, Pred RMSE: {rmse_pred:.8f})', 
                        fontsize=12)
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel(self.state_names[i], fontsize=10)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_multi_step_comparison(self, all_pred, all_base, all_gt, save_path='multi_step_comparison.png'):
        """绘制多个样本的预测对比图"""
        save_path = self._get_save_path(save_path)
        
        # all_pred, all_base, all_gt: (num_samples, state_dim)
        # each is 2D array with shape (N, 4)
        num_samples = all_pred.shape[0]
        
        # 创建4个子图，分别显示vlon, vlat, yaw, omega
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        state_labels = ['vlon (m/s)', 'vlat (m/s)', 'yaw (rad)', 'omega (rad/s)']
        
        for i in range(4):
            ax = axes[i]
            
            # 绘制所有样本的点
            ax.plot(range(num_samples), all_gt[:, i], label='Ground Truth (Real)', 
                    color='#27AE60', linewidth=2, linestyle='-', marker='o', markersize=4)
            ax.plot(range(num_samples), all_base[:, i], label='Physics Baseline', 
                    color='#3498DB', linewidth=2, linestyle='--', marker='s', markersize=4)
            ax.plot(range(num_samples), all_pred[:, i], label='Hybrid Corrected', 
                    color='#E74C3C', linewidth=2, linestyle='-.', marker='^', markersize=4)
            
            # 计算RMSE
            rmse_base = np.sqrt(np.mean((all_base[:, i] - all_gt[:, i])**2))
            rmse_pred = np.sqrt(np.mean((all_pred[:, i] - all_gt[:, i])**2))
            
            ax.set_title(f'{state_labels[i]}\nBase RMSE: {rmse_base:.8f}, Hybrid RMSE: {rmse_pred:.8f}', 
                        fontsize=11)
            ax.set_xlabel('Sample Index', fontsize=10)
            ax.set_ylabel(state_labels[i], fontsize=10)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.suptitle(f'Validation Prediction Comparison ({num_samples} samples)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_xy_trajectory(self, gt_xy, base_xy, pred_xy, save_path='xy_trajectory.png'):
        save_path = self._get_save_path(save_path)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], label='Ground Truth', 
                color='#2ECC71', linewidth=2, linestyle='-')
        ax.plot(base_xy[:, 0], base_xy[:, 1], label='Physics Baseline', 
                color='#3498DB', linewidth=2, linestyle='--')
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], label='Hybrid Corrected', 
                color='#E74C3C', linewidth=2, linestyle='-')
        
        ax.scatter(gt_xy[0, 0], gt_xy[0, 1], color='#F39C12', marker='*', s=150, label='Start')
        ax.scatter(gt_xy[-1, 0], gt_xy[-1, 1], color='#9B59B6', marker='s', s=100, label='End')
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('XY Trajectory Comparison', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axis('equal')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_error_over_time(self, base_error, pred_error, save_path='error_over_time.png'):
        save_path = self._get_save_path(save_path)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        time_steps = np.arange(base_error.shape[0])

        for i in range(4):
            ax = axes[i]

            ax.plot(time_steps, np.abs(base_error[:, i]), label='Base Error',
                    color='#3498DB', linewidth=2, linestyle='--')
            ax.plot(time_steps, np.abs(pred_error[:, i]), label='Pred Error',
                    color='#E74C3C', linewidth=2, linestyle='-')

            mean_base_error = np.mean(np.abs(base_error[:, i]))
            mean_pred_error = np.mean(np.abs(pred_error[:, i]))

            ax.set_title(f'{self.state_names[i]} Error (Base: {mean_base_error:.8f}, Pred: {mean_pred_error:.8f})',
                        fontsize=12)
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel('Absolute Error', fontsize=10)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_inference_results(self, state_data, physics_pred, hybrid_pred, 
                               save_path='inference_result.png', title='Inference Results'):
        """绘制推理结果：状态对比和误差分析"""
        save_path = self._get_save_path(save_path)
        
        fig = plt.figure(figsize=(20, 15))
        
        # 子图1-4：状态对比
        for i in range(4):
            ax = fig.add_subplot(3, 2, i+1)
            time_steps = np.arange(len(state_data))
            
            ax.plot(time_steps, state_data[:, i], label='Ground Truth', 
                    color='#27AE60', linewidth=2, linestyle='-')
            ax.plot(time_steps, physics_pred[:, i], label='Physics Baseline', 
                    color='#3498DB', linewidth=2, linestyle='--')
            ax.plot(time_steps, hybrid_pred[:, i], label='Hybrid Corrected', 
                    color='#E74C3C', linewidth=2, linestyle='-.')
            
            rmse_base = np.sqrt(np.mean((physics_pred[:, i] - state_data[:, i])**2))
            rmse_hybrid = np.sqrt(np.mean((hybrid_pred[:, i] - state_data[:, i])**2))
            
            ax.set_title(f'{self.state_names[i]}\nBase RMSE: {rmse_base:.8f}, Hybrid RMSE: {rmse_hybrid:.8f}', 
                        fontsize=12)
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel(self.state_names[i], fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # 子图5：误差对比
        ax5 = fig.add_subplot(3, 2, 5)
        base_error = np.mean(np.abs(physics_pred - state_data), axis=1)
        hybrid_error = np.mean(np.abs(hybrid_pred - state_data), axis=1)
        time_steps = np.arange(len(state_data))
        
        ax5.plot(time_steps, base_error, label='Base Error', 
                 color='#3498DB', linewidth=2, linestyle='--')
        ax5.plot(time_steps, hybrid_error, label='Hybrid Error', 
                 color='#E74C3C', linewidth=2, linestyle='-.')
        
        ax5.set_title(f'Mean Absolute Error Comparison', fontsize=12)
        ax5.set_xlabel('Time Step', fontsize=10)
        ax5.set_ylabel('MAE', fontsize=10)
        ax5.legend(fontsize=9)
        ax5.grid(True, linestyle='--', alpha=0.7)
        ax5.set_ylim(bottom=0)
        
        # 子图6：改进幅度
        ax6 = fig.add_subplot(3, 2, 6)
        improvement = base_error - hybrid_error
        ax6.plot(time_steps, improvement, label='Improvement (Base - Hybrid)', 
                 color='#2ECC71', linewidth=2)
        ax6.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        mean_improvement = np.mean(improvement)
        ax6.set_title(f'Improvement (Mean: {mean_improvement:.8f})', fontsize=12)
        ax6.set_xlabel('Time Step', fontsize=10)
        ax6.set_ylabel('Improvement', fontsize=10)
        ax6.legend(fontsize=9)
        ax6.grid(True, linestyle='--', alpha=0.7)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_xy_comparison(self, gt_xy, physics_xy, hybrid_xy, 
                           save_path='xy_comparison.png', title='XY Trajectory Comparison'):
        """绘制XY轨迹对比图"""
        save_path = self._get_save_path(save_path)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], label='Ground Truth', 
                color='#27AE60', linewidth=3, linestyle='-', alpha=0.9)
        ax.plot(physics_xy[:, 0], physics_xy[:, 1], label='Physics Baseline', 
                color='#3498DB', linewidth=2.5, linestyle='--', alpha=0.8)
        ax.plot(hybrid_xy[:, 0], hybrid_xy[:, 1], label='Hybrid Corrected', 
                color='#E74C3C', linewidth=2.5, linestyle='-.', alpha=0.8)
        
        # 标记起点和终点
        ax.scatter(gt_xy[0, 0], gt_xy[0, 1], color='#F39C12', marker='*', s=200, zorder=10, label='Start')
        ax.scatter(gt_xy[-1, 0], gt_xy[-1, 1], color='#9B59B6', marker='s', s=120, zorder=10, label='End')
        
        # 标记关键帧
        step = max(1, len(gt_xy) // 10)
        for i in range(0, len(gt_xy), step):
            ax.scatter(gt_xy[i, 0], gt_xy[i, 1], color='#27AE60', marker='o', s=30, alpha=0.5)
        
        # 计算轨迹误差
        phy_dist = np.sqrt((gt_xy[:, 0] - physics_xy[:, 0])**2 + (gt_xy[:, 1] - physics_xy[:, 1])**2)
        hybrid_dist = np.sqrt((gt_xy[:, 0] - hybrid_xy[:, 0])**2 + (gt_xy[:, 1] - hybrid_xy[:, 1])**2)
        
        rmse_phy = np.sqrt(np.mean(phy_dist**2))
        rmse_hybrid = np.sqrt(np.mean(hybrid_dist**2))
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'{title}\nBase RMSE: {rmse_phy:.6f}m, Hybrid RMSE: {rmse_hybrid:.6f}m', fontsize=14)
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axis('equal')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_4trajectories(self, gt_xy, physics_xy, hybrid_interactive_xy, hybrid_additive_xy,
                           save_path='4trajectories.png'):
        save_path = self._get_save_path(save_path)

        fig, ax = plt.subplots(figsize=(12, 9))

        ax.plot(gt_xy[:, 0], gt_xy[:, 1], label='Ground Truth',
                color='#2ECC71', linewidth=2.5, linestyle='-', marker='', alpha=0.9)
        ax.plot(physics_xy[:, 0], physics_xy[:, 1], label='Physics Baseline (Open-Loop)',
                color='#3498DB', linewidth=2, linestyle='--', alpha=0.8)
        ax.plot(hybrid_interactive_xy[:, 0], hybrid_interactive_xy[:, 1], label='Hybrid Interactive (Closed-Loop)',
                color='#E74C3C', linewidth=2, linestyle='-', alpha=0.8)
        ax.plot(hybrid_additive_xy[:, 0], hybrid_additive_xy[:, 1], label='Hybrid Additive (Open-Loop)',
                color='#9B59B6', linewidth=2, linestyle='-.', alpha=0.8)

        ax.scatter(gt_xy[0, 0], gt_xy[0, 1], color='#F39C12', marker='*', s=200, zorder=10, label='Start')
        ax.scatter(gt_xy[-1, 0], gt_xy[-1, 1], color='#1ABC9C', marker='s', s=120, zorder=10, label='End')

        for i in range(0, len(gt_xy), max(1, len(gt_xy) // 5)):
            ax.scatter(gt_xy[i, 0], gt_xy[i, 1], color='#2ECC71', marker='o', s=30, alpha=0.5, zorder=5)

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Trajectory Comparison: Physics vs Hybrid Models', fontsize=14)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axis('equal')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_model_comparison(self, physics_preds, hybrid_preds, state_data, 
                                   start_idx=0, hist_len=15, save_path='state_comparison.png'):
        """状态对比可视化"""
        save_path = self._get_save_path(save_path)
        
        compare_start = start_idx + hist_len
        n_states = min(4, physics_preds.shape[1])
        
        fig, axes = plt.subplots(n_states, 1, figsize=(14, 10))
        if n_states == 1:
            axes = [axes]
        
        time_steps = np.arange(len(state_data))
        
        for i in range(n_states):
            ax = axes[i]
            ax.plot(time_steps, state_data[:, i], label='Ground Truth', 
                    color='#2ECC71', linewidth=2, alpha=0.9)
            ax.plot(time_steps, physics_preds[:, i], label='Physics Baseline', 
                    color='#3498DB', linewidth=1.5, linestyle='--', alpha=0.8)
            ax.plot(time_steps, hybrid_preds[:, i], label='Hybrid Prediction', 
                    color='#E74C3C', linewidth=1.5, linestyle='-.', alpha=0.8)
            
            ax.axvline(x=compare_start, color='black', linestyle='--', alpha=0.5, label='Prediction Start')
            
            rmse_base = np.sqrt(np.mean((physics_preds[compare_start:, i] - state_data[compare_start:, i])**2))
            rmse_hybrid = np.sqrt(np.mean((hybrid_preds[compare_start:, i] - state_data[compare_start:, i])**2))
            
            ax.set_title(f'{self.state_names[i]} (Base RMSE: {rmse_base:.6f}, Hybrid RMSE: {rmse_hybrid:.6f})')
            ax.set_xlabel('Time Step')
            ax.set_ylabel(self.state_names[i])
            ax.legend(fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_error_curves(self, cumulative_errors, start_idx=0, hist_len=15, 
                          save_path='error_curves.png'):
        """绘制累积误差曲线"""
        save_path = self._get_save_path(save_path)
        
        compare_start = start_idx + hist_len
        n_features = min(4, cumulative_errors.shape[1])
        
        fig, axes = plt.subplots(n_features, 1, figsize=(14, 10))
        if n_features == 1:
            axes = [axes]
        
        time_steps = np.arange(len(cumulative_errors))
        
        for i in range(n_features):
            ax = axes[i]
            ax.plot(time_steps, cumulative_errors[:, i], label=f'{self.state_names[i]} Error',
                    color='#E74C3C', linewidth=1.5)
            
            rmse = np.sqrt(np.mean(cumulative_errors[:, i]**2))
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.axhline(y=rmse, color='#F39C12', linestyle=':', alpha=0.7, label=f'RMSE: {rmse:.6f}')
            ax.axhline(y=-rmse, color='#F39C12', linestyle=':', alpha=0.7)
            
            ax.set_title(f'{self.state_names[i]} Cumulative Error')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Error')
            ax.legend(fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_data_comparison(self, state_data, base_data, control_data, save_path='./'):
        """数据对比可视化"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 状态对比
        self.plot_state_comparison(state_data, base_data, state_data, 
                                   save_path=os.path.join(save_path, 'state_vs_base.png'))
        
        # 控制输入可视化
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
        """3D散点图可视化"""
        if save_path is None:
            save_path = self.save_dir
        os.makedirs(save_path, exist_ok=True)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for res in results:
            gt_xy = res['targets'][:, :2] if res['targets'].shape[1] >= 2 else res['physics_xy_full']
            ax.scatter(gt_xy[:, 0], gt_xy[:, 1], np.arange(len(gt_xy)), 
                       c='green', marker='o', alpha=0.5, label='Ground Truth')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Time Step')
        ax.set_title('3D Trajectory View')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, '3d_trajectory.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_error_comparison(self, results, save_path=None):
        """误差对比可视化"""
        if save_path is None:
            save_path = self.save_dir
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
        """误差与控制角度关系可视化"""
        if save_path is None:
            save_path = self.save_dir
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
        """XY重建对比可视化"""
        if save_path is None:
            save_path = self.save_dir
        os.makedirs(save_path, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        for res in results:
            gt_xy = res['physics_xy_full']  # 假设physics_xy_full包含真实XY
            phy_xy = res['physics_xy_full']
            hyb_xy = res['hybrid_xy_full']
            
            axes[0].plot(gt_xy[:, 0], gt_xy[:, 1], color='#2ECC71', linewidth=1, alpha=0.3)
            axes[1].plot(phy_xy[:, 0], phy_xy[:, 1], color='#3498DB', linewidth=1, alpha=0.3)
            axes[1].plot(hyb_xy[:, 0], hyb_xy[:, 1], color='#E74C3C', linewidth=1, alpha=0.3)
        
        axes[0].set_title('Ground Truth Trajectories')
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].grid(True, linestyle='--', alpha=0.7)
        axes[0].axis('equal')
        
        axes[1].set_title('Physics vs Hybrid Trajectories')
        axes[1].set_xlabel('X (m)')
        axes[1].set_ylabel('Y (m)')
        axes[1].legend(['Physics', 'Hybrid'])
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].axis('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'xy_reconstruction.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_xy_comparison_fixed_gt(self, results, save_path=None):
        """固定GT的XY对比可视化"""
        if save_path is None:
            save_path = self.save_dir
        os.makedirs(save_path, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        for res in results:
            gt_xy = res['physics_xy_full']
            hyb_xy = res['hybrid_xy_full']
            
            # 归一化到起点
            start_x, start_y = gt_xy[0, 0], gt_xy[0, 1]
            gt_normalized = gt_xy - np.array([start_x, start_y])
            hyb_normalized = hyb_xy - np.array([start_x, start_y])
            
            ax.plot(gt_normalized[:, 0], gt_normalized[:, 1], color='#2ECC71', linewidth=1.5, alpha=0.5)
            ax.plot(hyb_normalized[:, 0], hyb_normalized[:, 1], color='#E74C3C', linewidth=1.5, alpha=0.5)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('XY Trajectory Comparison (Normalized to Start)')
        ax.legend(['Ground Truth', 'Hybrid'])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'xy_fixed_gt.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_fluctuation_analysis(self, state_data, physics_pred, hybrid_pred, 
                                  threshold_frame=400, save_path='fluctuation_analysis.png'):
        """分析400帧后波动过大问题"""
        save_path = self._get_save_path(save_path)
        
        fig = plt.figure(figsize=(20, 12))
        
        n_states = min(4, state_data.shape[1])
        
        for i in range(n_states):
            ax = fig.add_subplot(n_states, 1, i+1)
            
            # 全序列
            time_steps = np.arange(len(state_data))
            ax.plot(time_steps, state_data[:, i], label='Ground Truth', 
                    color='#2ECC71', linewidth=1.5, alpha=0.9)
            ax.plot(time_steps, physics_pred[:, i], label='Physics', 
                    color='#3498DB', linewidth=1.5, linestyle='--', alpha=0.8)
            ax.plot(time_steps, hybrid_pred[:, i], label='Hybrid', 
                    color='#E74C3C', linewidth=1.5, linestyle='-.', alpha=0.8)
            
            # 标记400帧位置
            if threshold_frame < len(state_data):
                ax.axvline(x=threshold_frame, color='#F39C12', linestyle='--', linewidth=2, 
                          label=f'{threshold_frame} Frame')
            
            ax.set_title(f'{self.state_names[i]} - Full Sequence')
            ax.set_xlabel('Time Step')
            ax.set_ylabel(self.state_names[i])
            ax.legend(fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_fluctuation_detailed(self, state_data, physics_pred, hybrid_pred, 
                                  start_frame=400, window_size=200, save_path='fluctuation_detail.png'):
        """详细分析指定帧范围的波动"""
        save_path = self._get_save_path(save_path)
        
        end_frame = min(start_frame + window_size, len(state_data))
        time_window = np.arange(start_frame, end_frame)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        n_states = min(4, state_data.shape[1])
        
        for i in range(n_states):
            ax = axes[i]
            
            ax.plot(time_window, state_data[start_frame:end_frame, i], 
                    label='Ground Truth', color='#2ECC71', linewidth=2)
            ax.plot(time_window, physics_pred[start_frame:end_frame, i], 
                    label='Physics', color='#3498DB', linewidth=1.5, linestyle='--')
            ax.plot(time_window, hybrid_pred[start_frame:end_frame, i], 
                    label='Hybrid', color='#E74C3C', linewidth=1.5, linestyle='-.')
            
            # 计算波动指标
            gt_std = np.std(state_data[start_frame:end_frame, i])
            phy_std = np.std(physics_pred[start_frame:end_frame, i])
            hyb_std = np.std(hybrid_pred[start_frame:end_frame, i])
            
            ax.set_title(f'{self.state_names[i]} (Frames {start_frame}-{end_frame})\n'
                        f'STD: GT={gt_std:.6f}, Physics={phy_std:.6f}, Hybrid={hyb_std:.6f}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel(self.state_names[i])
            ax.legend(fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    vis = VisualUtils(state_names=['vlon', 'vlat', 'yaw', 'omega'], save_dir='./plots')
    
    np.random.seed(42)
    train_losses = np.random.rand(50) * 0.5 + 0.5
    train_losses = np.sort(train_losses)[::-1]
    val_losses = train_losses * (0.9 + np.random.rand(50) * 0.2)
    vis.plot_loss_curve(train_losses, val_losses, 'test_loss.png')
    
    T = 100
    gt_states = np.random.randn(T, 4) * 0.5 + np.array([20, 0.5, 0.1, 0.05])
    base_states = gt_states + np.random.randn(T, 4) * 0.3
    pred_states = gt_states + np.random.randn(T, 4) * 0.1
    vis.plot_state_comparison(pred_states, base_states, gt_states, 'test_state.png')
    
    t = np.linspace(0, 10, T)
    gt_xy = np.column_stack([20 * t, 5 * np.sin(0.5 * t)])
    base_xy = gt_xy + np.random.randn(T, 2) * 0.3
    pred_xy = gt_xy + np.random.randn(T, 2) * 0.1
    vis.plot_xy_trajectory(gt_xy, base_xy, pred_xy, 'test_trajectory.png')
    
    base_error = np.random.randn(T, 4) * 0.3
    pred_error = np.random.randn(T, 4) * 0.1
    vis.plot_error_over_time(base_error, pred_error, 'test_error.png')
    
    print("All test plots saved successfully!")
