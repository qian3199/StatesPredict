import matplotlib.pyplot as plt
import numpy as np
import os


class VisualUtils:
    def __init__(self, state_names=None, save_dir='./plots'):
        self.state_names = state_names if state_names else ['vlon', 'vlat', 'yaw', 'omega']
        self.save_dir = save_dir
        
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
            
            ax.set_title(f'{self.state_names[i]} (Base RMSE: {rmse_base:.4f}, Pred RMSE: {rmse_pred:.4f})', 
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
            
            ax.set_title(f'{state_labels[i]}\nBase RMSE: {rmse_base:.4f}, Hybrid RMSE: {rmse_pred:.4f}', 
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

            ax.set_title(f'{self.state_names[i]} Error (Base: {mean_base_error:.4f}, Pred: {mean_pred_error:.4f})',
                        fontsize=12)
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel('Absolute Error', fontsize=10)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_ylim(bottom=0)

        plt.tight_layout()
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
