# tools/VisualUtils.py
import os
import matplotlib.pyplot as plt
import numpy as np

class VisualUtils:
    def __init__(self, state_names, save_dir):
        """
        :param state_names: 状态名称列表（如 ['v_lon', 'v_lat', 'yaw', 'v_yaw']）
        :param save_dir: 图片保存目录
        """
        self.state_names = state_names
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def visual_loss_function(self, early_step, train_losses, val_losses, label='training_val_loss'):
        """
        绘制训练和验证损失曲线
        :param early_step: 保留参数（与原接口一致，未使用）
        :param train_losses: 训练损失列表
        :param val_losses: 验证损失列表
        :param label: 保存图片的文件名前缀
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Train Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_path = os.path.join(self.save_dir, f'{label}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Loss figure saved to {save_path}")

    def plot_trajectory_comparison(self, real_x, real_y, pred_x, pred_y, title='Trajectory Comparison', filename='trajectory.png'):
        """绘制轨迹对比图（真实 vs 预测）"""
        plt.figure(figsize=(10, 10))
        plt.plot(real_x, real_y, 'g-', linewidth=2, label='Real')
        plt.plot(pred_x, pred_y, 'b--', linewidth=2, label='Predicted')
        plt.title(title)
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.axis('equal')
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Trajectory figure saved to {save_path}")

    def plot_state_comparison(self, real_states, pred_states, time=None, title_prefix='State', filename='states_comparison.png'):
        """
        绘制各状态的时间序列对比（每个状态一个子图）
        输入形状：(seq_len, state_dim) 或 (state_dim, seq_len) 均可自动适配
        """
        # 自动转置为 (seq_len, state_dim)
        if real_states.ndim == 2 and real_states.shape[0] < real_states.shape[1]:
            real_states = real_states.T
            pred_states = pred_states.T
        seq_len, state_dim = real_states.shape
        if time is None:
            time = np.arange(seq_len)
        n_cols = min(2, state_dim)
        n_rows = (state_dim + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten() if state_dim > 1 else [axes]
        for i in range(state_dim):
            ax = axes[i]
            ax.plot(time, real_states[:, i], 'g-', linewidth=2, label='Real')
            ax.plot(time, pred_states[:, i], 'b--', linewidth=2, label='Predicted')
            name = self.state_names[i] if i < len(self.state_names) else f"State {i}"
            ax.set_title(f'{name}')
            ax.set_xlabel('Time step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        for i in range(state_dim, len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"State comparison figure saved to {save_path}")

    def plot_error(self, errors, state_names=None, title='Prediction Error', filename='error.png'):
        """绘制各状态误差曲线，并标注RMSE"""
        if state_names is None:
            state_names = self.state_names
        seq_len, state_dim = errors.shape
        n_cols = min(2, state_dim)
        n_rows = (state_dim + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten() if state_dim > 1 else [axes]
        for i in range(state_dim):
            ax = axes[i]
            ax.plot(errors[:, i], 'r-', linewidth=1)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            name = state_names[i] if i < len(state_names) else f"Dim {i}"
            ax.set_title(f'Error: {name}')
            ax.set_xlabel('Time step')
            ax.set_ylabel('Error')
            ax.grid(True, alpha=0.3)
            rmse = np.sqrt(np.mean(errors[:, i]**2))
            ax.text(0.02, 0.95, f'RMSE: {rmse:.4f}', transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        for i in range(state_dim, len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Error figure saved to {save_path}")
        
    def visual_res_function(self, epoch, gt, base_state, model_state, base_error, model_error,
                            label=None, save_results=True):
        """可视化残差对比图（兼容 Tensor 和 numpy）"""
        # 1. 强制转换为 numpy 数组（若为 Tensor）
        if hasattr(base_error, 'numpy'):
            base_error = base_error.detach().cpu().numpy()
        if hasattr(model_error, 'numpy'):
            model_error = model_error.detach().cpu().numpy()
        
        # 2. 处理 3D 数据 [batch, seq, dim]
        if base_error.ndim == 3:
            print(f"Warning: base_error is 3D with shape {base_error.shape}, using first sample (batch=0).")
            base_error = base_error[0]
            model_error = model_error[0]
        
        # 3. 确保形状为 (seq_len, dim)
        if base_error.ndim == 2 and base_error.shape[0] < base_error.shape[1]:
            base_error = base_error.T
            model_error = model_error.T
        
        seq_len, state_dim = base_error.shape
        
        # 4. 创建子图
        n_cols = min(2, state_dim)
        n_rows = (state_dim + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        if state_dim == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # 5. 绘图
        for i in range(state_dim):
            ax = axes[i]
            ax.plot(base_error[:, i], 'r-', linewidth=1.5, label='Base Model Error')
            ax.plot(model_error[:, i], 'b--', linewidth=1.5, label='Our Model Error')
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            name = self.state_names[i] if i < len(self.state_names) else f"Dim {i}"
            ax.set_title(f'Error: {name}')
            ax.set_xlabel('Time step')
            ax.set_ylabel('Error')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # RMSE (此时 base_error 已是 numpy 数组)
            base_rmse = np.sqrt(np.mean(base_error[:, i] ** 2))
            model_rmse = np.sqrt(np.mean(model_error[:, i] ** 2))
            ax.text(0.02, 0.95, f'Base RMSE: {base_rmse:.4f}\nOur RMSE: {model_rmse:.4f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 隐藏多余子图
        for i in range(state_dim, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        if save_results:
            if label is None:
                label = f"res_epoch_{epoch+1}"
            save_path = os.path.join(self.save_dir, f"{label}.png")
            plt.savefig(save_path, dpi=150)
            print(f"Residual figure saved to {save_path}")
        else:
            plt.show()
        plt.close()


    def visual_trajectory_function(self, gt, base_state, model_state, dt=0.01,
                                label=None, save_results=True, max_samples=None):
        """可视化轨迹对比（兼容 Tensor 和 numpy）"""
        # 1. 转换为 numpy
        if hasattr(gt, 'numpy'):
            gt = gt.detach().cpu().numpy()
        if hasattr(base_state, 'numpy'):
            base_state = base_state.detach().cpu().numpy()
        if hasattr(model_state, 'numpy'):
            model_state = model_state.detach().cpu().numpy()
        
        # 2. 处理 3D 数据
        if gt.ndim == 3:
            print(f"Warning: gt is 3D with shape {gt.shape}, using first sample (batch=0).")
            gt = gt[0]
            base_state = base_state[0]
            model_state = model_state[0]
        
        # 3. 确保形状为 (seq_len, dim)
        if gt.ndim == 2 and gt.shape[0] < gt.shape[1]:
            gt = gt.T
            base_state = base_state.T
            model_state = model_state.T
        
        # 4. 采样（可选）
        seq_len = gt.shape[0]
        if max_samples is not None and max_samples < seq_len:
            indices = np.linspace(0, seq_len - 1, max_samples, dtype=int)
            gt = gt[indices]
            base_state = base_state[indices]
            model_state = model_state[indices]
        
        # 5. 绘图
        plt.figure(figsize=(10, 10))
        plt.plot(gt[:, 0], gt[:, 1], 'g-', linewidth=2.5, label='Ground Truth')
        plt.plot(base_state[:, 0], base_state[:, 1], 'r--', linewidth=2, label='Base Model')
        plt.plot(model_state[:, 0], model_state[:, 1], 'b-.', linewidth=2, label='Our Model')
        plt.title('Trajectory Comparison')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.axis('equal')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_results:
            if label is None:
                label = "trajectory_comparison"
            save_path = os.path.join(self.save_dir, f"{label}.png")
            plt.savefig(save_path, dpi=150)
            print(f"Trajectory figure saved to {save_path}")
        else:
            plt.show()
        plt.close()