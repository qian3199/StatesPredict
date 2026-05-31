import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import logging
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, state_scaler, diff_scaler, lr=1e-4, weight_decay=1e-5, visualizer=None, log_dir='./logs', device=None):
        self.model = model
        self.state_scaler = state_scaler
        self.diff_scaler = diff_scaler
        self.visualizer = visualizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)

        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.train_losses = []
        self.val_losses = []

    def denormalize_state(self, state_norm):
        # 使用tensor操作来保留梯度，不转numpy
        mean = torch.FloatTensor(self.state_scaler.mean_).to(state_norm.device)
        scale = torch.FloatTensor(self.state_scaler.scale_).to(state_norm.device)
        
        if state_norm.ndim == 3:
            return state_norm * scale.unsqueeze(0).unsqueeze(0) + mean.unsqueeze(0).unsqueeze(0)
        else:
            return state_norm * scale + mean

    def denormalize_diff(self, diff_norm):
        # 使用tensor操作来保留梯度，不转numpy
        mean = torch.FloatTensor(self.diff_scaler.mean_).to(diff_norm.device)
        scale = torch.FloatTensor(self.diff_scaler.scale_).to(diff_norm.device)
        
        if diff_norm.ndim == 3:
            return diff_norm * scale.unsqueeze(0).unsqueeze(0) + mean.unsqueeze(0).unsqueeze(0)
        else:
            return diff_norm * scale + mean

    def compute_trajectory_loss(self, states_pred, states_gt, dt=0.01):
        """计算XY轨迹误差（从状态积分出轨迹后计算）"""
        batch_size, seq_len, _ = states_pred.shape
        
        # 积分轨迹
        x_pred = torch.zeros(batch_size, seq_len).to(states_pred.device)
        y_pred = torch.zeros(batch_size, seq_len).to(states_pred.device)
        x_gt = torch.zeros(batch_size, seq_len).to(states_pred.device)
        y_gt = torch.zeros(batch_size, seq_len).to(states_pred.device)
        
        for i in range(1, seq_len):
            vlon_pred = states_pred[:, i-1, 0]
            vlat_pred = states_pred[:, i-1, 1]
            yaw_pred = states_pred[:, i-1, 2]
            
            vlon_gt = states_gt[:, i-1, 0]
            vlat_gt = states_gt[:, i-1, 1]
            yaw_gt = states_gt[:, i-1, 2]
            
            dx_pred = (vlon_pred * torch.cos(yaw_pred) - vlat_pred * torch.sin(yaw_pred)) * dt
            dy_pred = (vlon_pred * torch.sin(yaw_pred) + vlat_pred * torch.cos(yaw_pred)) * dt
            
            dx_gt = (vlon_gt * torch.cos(yaw_gt) - vlat_gt * torch.sin(yaw_gt)) * dt
            dy_gt = (vlon_gt * torch.sin(yaw_gt) + vlat_gt * torch.cos(yaw_gt)) * dt
            
            x_pred[:, i] = x_pred[:, i-1] + dx_pred
            y_pred[:, i] = y_pred[:, i-1] + dy_pred
            
            x_gt[:, i] = x_gt[:, i-1] + dx_gt
            y_gt[:, i] = y_gt[:, i-1] + dy_gt
        
        # 计算轨迹误差
        traj_error = torch.sqrt((x_pred - x_gt)**2 + (y_pred - y_gt)**2)
        return traj_error.mean()

    def compute_cascade_loss(self, corrected_norm, gt_norm, base_norm, delta_norm, weights=[0.1, 0.1, 0.1, 0.1], warm_up=0):
        corrected_real = self.denormalize_state(corrected_norm)
        gt_real = self.denormalize_state(gt_norm)
        base_real = self.denormalize_state(base_norm)
        delta_real = self.denormalize_diff(delta_norm)

        final_pred = base_real + delta_real
        
        pred_loss = torch.abs(final_pred - gt_real).mean()
        traj_loss = self.compute_trajectory_loss(final_pred, gt_real)
        delta_reg_loss = torch.abs(delta_real).mean()
        
        base_error = torch.abs(base_real - gt_real).mean()
        corr_error = torch.abs(final_pred - gt_real).mean()
        diff_constraint = torch.max(torch.tensor(0.0, device=final_pred.device), corr_error - base_error)

        cascade_loss = pred_loss * 5.0 + traj_loss * 20.0 + delta_reg_loss * 5.0 + diff_constraint * 50.0
        
        mse_loss = torch.mean((final_pred - gt_real) ** 2)
        
        gt_mean = torch.mean(gt_real, dim=1, keepdim=True)
        ss_tot = torch.sum((gt_real - gt_mean) ** 2, dim=[1, 2])
        ss_res = torch.sum((final_pred - gt_real) ** 2, dim=[1, 2])
        r2_score = torch.mean(1 - ss_res / (ss_tot + 1e-8))
        
        base_mse = torch.mean((base_real - gt_real) ** 2)
        pred_mse = torch.mean((final_pred - gt_real) ** 2)
        
        return cascade_loss, pred_loss, traj_loss, mse_loss, r2_score, base_mse, pred_mse

    def train_step(self, hist_state, hist_control, base_norm, cfg_tensor, gt_norm, diff_norm):
        self.model.train()
        self.optimizer.zero_grad()
        
        hist_state = hist_state.to(self.device)
        hist_control = hist_control.to(self.device)
        base_norm = base_norm.to(self.device)
        cfg_tensor = cfg_tensor.to(self.device)
        gt_norm = gt_norm.to(self.device)
        diff_norm = diff_norm.to(self.device)

        delta_norm, corrected_norm = self.model(hist_state, hist_control, base_norm, cfg_tensor, gt_norm)

        cascade_loss, v_lat_loss, ade_loss, mse_loss, r2_score, base_mse, pred_mse = self.compute_cascade_loss(
            corrected_norm, gt_norm, base_norm, delta_norm, weights=[0.1, 0.1, 0.1, 0.1], warm_up=0
        )

        delta_mse = self.mse_loss(delta_norm, diff_norm).mean()

        total_loss = cascade_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'cascade_loss': cascade_loss.item(),
            'v_lat_loss': v_lat_loss.item(),
            'ade_loss': ade_loss.item(),
            'mse_loss': mse_loss.item(),
            'r2_score': r2_score.item(),
            'base_mse': base_mse.item(),
            'pred_mse': pred_mse.item(),
            'delta_mse': delta_mse.item()
        }

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_cascade = 0.0
        total_mse = 0.0
        total_r2 = 0.0
        total_base_mse = 0.0
        total_pred_mse = 0.0
        count = 0
        val_batch_data = None

        with torch.no_grad():
            for i, (hist_state, hist_control, base_norm, cfg_tensor, gt_norm, diff_norm, future_pos) in enumerate(val_loader):
                hist_state = hist_state.to(self.device)
                hist_control = hist_control.to(self.device)
                base_norm = base_norm.to(self.device)
                cfg_tensor = cfg_tensor.to(self.device)
                gt_norm = gt_norm.to(self.device)
                diff_norm = diff_norm.to(self.device)
                
                delta_norm, corrected_norm = self.model(hist_state, hist_control, base_norm, cfg_tensor)

                cascade_loss, _, _, mse_loss, r2_score, base_mse, pred_mse = self.compute_cascade_loss(
                    corrected_norm, gt_norm, base_norm, delta_norm, weights=[0.1, 0.1, 0.1, 0.1], warm_up=0
                )

                total_loss += cascade_loss.item()
                total_cascade += cascade_loss.item()
                total_mse += mse_loss.item()
                total_r2 += r2_score.item()
                total_base_mse += base_mse.item()
                total_pred_mse += pred_mse.item()
                count += 1

                if i == 0:
                    val_batch_data = {
                        'hist_state': hist_state,
                        'hist_control': hist_control,
                        'base_norm': base_norm,
                        'cfg_tensor': cfg_tensor,
                        'gt_norm': gt_norm,
                        'diff_norm': diff_norm,
                        'corrected_norm': corrected_norm,
                        'delta_norm': delta_norm,
                        'future_pos': future_pos
                    }

        return {
            'val_loss': total_loss / count,
            'val_cascade': total_cascade / count,
            'val_mse': total_mse / count,
            'val_r2': total_r2 / count,
            'val_base_mse': total_base_mse / count,
            'val_pred_mse': total_pred_mse / count,
            'val_batch_data': val_batch_data
        }

    def integrate_trajectory(self, states, dt=0.01):
        """从状态积分出xy轨迹"""
        n = states.shape[0]
        x = np.zeros(n)
        y = np.zeros(n)
        
        for i in range(1, n):
            vlon = states[i-1, 0]
            vlat = states[i-1, 1]
            yaw = states[i-1, 2]
            
            dx = (vlon * np.cos(yaw) - vlat * np.sin(yaw)) * dt
            dy = (vlon * np.sin(yaw) + vlat * np.cos(yaw)) * dt
            
            x[i] = x[i-1] + dx
            y[i] = y[i-1] + dy
        
        return np.column_stack([x, y])

    def autoregressive_prediction(self, hist_state, hist_control, base_states, steps=20):
        """使用单步模型做多步自回归预测"""
        self.model.eval()
        
        batch_size = hist_state.size(0)
        device = hist_state.device
        
        # 获取初始特征
        hist_input = torch.cat([hist_state, hist_control], dim=-1)
        features = self.model.feature_mlp(hist_input)
        _, (h_n, _) = self.model.lstm(features)
        h_n = h_n[-1]
        
        # 初始化
        current_state = base_states[:, 0, :]
        cfg = torch.FloatTensor([1, 1, 0]).repeat(batch_size, 1).to(device)
        
        pred_states = []
        base_preds = []
        
        for t in range(steps):
            # 获取当前base预测
            if t < base_states.size(1):
                base_step = base_states[:, t, :]
            else:
                base_step = base_states[:, -1, :]
            
            base_preds.append(base_step)
            
            # 预测delta（在base_step上计算，保持与训练一致）
            mlp_input = torch.cat([h_n, base_step, cfg], dim=-1)
            delta = self.model.residual_mlp(mlp_input)
            
            # 使用tanh限制delta大小，防止突变
            delta_clamped = torch.tanh(delta) * 0.5  # 限制delta范围
            
            # 残差网络：直接 base + delta，不加多余的混合权重
            final_pred = base_step + delta_clamped
            
            pred_states.append(final_pred)
            current_state = final_pred.detach()
        
        return torch.stack(pred_states, dim=1), torch.stack(base_preds, dim=1)

    def plot_validation_comparison(self, val_batch_data, epoch):
        if self.visualizer is None:
            return

        self.model.eval()
        with torch.no_grad():
            delta_norm, corrected_norm = self.model(
                val_batch_data['hist_state'],
                val_batch_data['hist_control'],
                val_batch_data['base_norm'],
                val_batch_data['cfg_tensor']
            )

        base_real = self.denormalize_state(val_batch_data['base_norm'])
        gt_real = self.denormalize_state(val_batch_data['gt_norm'])
        delta_real = self.denormalize_diff(delta_norm)
        future_pos = val_batch_data['future_pos']

        base_pred = base_real + delta_real

        base_error = torch.sqrt((base_real - gt_real)**2).mean().item()
        corr_error = torch.sqrt((base_pred - gt_real)**2).mean().item()

        self.logger.info(f"  Base Error: {base_error:.6f} | Corrected Error: {corr_error:.6f} | Better: {corr_error < base_error}")

        batch_size = base_pred.size(0)
        num_samples = min(batch_size, 50)

        if base_pred.dim() == 3 and base_pred.size(1) == 1:
            base_pred_squeezed = base_pred.squeeze(1)
            gt_real_squeezed = gt_real.squeeze(1)
        else:
            base_pred_squeezed = base_pred
            gt_real_squeezed = gt_real

        all_pred_np = base_pred_squeezed[:num_samples].detach().cpu().numpy()
        all_base_np = base_real.squeeze(1)[:num_samples].detach().cpu().numpy()
        all_gt_np = gt_real_squeezed[:num_samples].detach().cpu().numpy()
        all_pos_np = future_pos[:num_samples].squeeze(1).detach().cpu().numpy()

        img_path = f'val_comparison_epoch{epoch:04d}.png'
        self.visualizer.plot_multi_step_comparison(all_pred_np, all_base_np, all_gt_np, img_path)

        traj_path = f'val_trajectory_epoch{epoch:04d}.png'
        fig, ax = plt.subplots(figsize=(12, 10))

        dt = 0.01
        real_yaw = all_gt_np[:, 2]

        dx_real = all_gt_np[:, 0] * np.cos(real_yaw) * dt - all_gt_np[:, 1] * np.sin(real_yaw) * dt
        dy_real = all_gt_np[:, 0] * np.sin(real_yaw) * dt + all_gt_np[:, 1] * np.cos(real_yaw) * dt

        base_yaw = all_base_np[:, 2]
        dx_base = all_base_np[:, 0] * np.cos(base_yaw) * dt - all_base_np[:, 1] * np.sin(base_yaw) * dt
        dy_base = all_base_np[:, 0] * np.sin(base_yaw) * dt + all_base_np[:, 1] * np.cos(base_yaw) * dt

        pred_yaw = all_pred_np[:, 2]
        dx_pred = all_pred_np[:, 0] * np.cos(pred_yaw) * dt - all_pred_np[:, 1] * np.sin(pred_yaw) * dt
        dy_pred = all_pred_np[:, 0] * np.sin(pred_yaw) * dt + all_pred_np[:, 1] * np.cos(pred_yaw) * dt

        real_x = all_pos_np[:, 0]
        real_y = all_pos_np[:, 1]
        base_x = real_x - dx_real + dx_base
        base_y = real_y - dy_real + dy_base
        pred_x = real_x - dx_real + dx_pred
        pred_y = real_y - dy_real + dy_pred

        sorted_idx = np.argsort(np.arctan2(real_y - np.mean(real_y), real_x - np.mean(real_x)))
        ax.plot(real_x[sorted_idx], real_y[sorted_idx], label='Ground Truth (Real)',
                color='#27AE60', linewidth=2, alpha=0.7, marker='o', markersize=3, zorder=3)
        ax.plot(base_x[sorted_idx], base_y[sorted_idx], label='Physics Baseline',
                color='#3498DB', linewidth=2, alpha=0.7, marker='s', markersize=3, zorder=2)
        ax.plot(pred_x[sorted_idx], pred_y[sorted_idx], label='Hybrid Corrected',
                color='#E74C3C', linewidth=2, alpha=0.7, marker='^', markersize=3, zorder=1)

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'Validation Position Comparison ({num_samples} samples) - Epoch {epoch}', fontsize=14)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axis('equal')
        plt.savefig(os.path.join(self.visualizer.save_dir, traj_path), dpi=150, bbox_inches='tight')
        plt.close()

    def train(self, train_loader, val_loader=None, epochs=100, log_interval=10):
        best_val_loss = float('inf')
        best_epoch = 0

        self.logger.info(f"Training started: {epochs} epochs, lr={self.optimizer.param_groups[0]['lr']}")

        for epoch in range(epochs):
            self.model.teacher_forcing_ratio = max(0.1, 0.5 - epoch * 0.004)

            train_losses = []
            for hist_state, hist_control, base_norm, cfg_tensor, gt_norm, diff_norm, _ in train_loader:
                losses = self.train_step(hist_state, hist_control, base_norm, cfg_tensor, gt_norm, diff_norm)
                train_losses.append(losses)

            avg_train_loss = np.mean([l['total_loss'] for l in train_losses])
            avg_cascade = np.mean([l['cascade_loss'] for l in train_losses])
            self.train_losses.append(avg_train_loss)

            is_best = False
            if val_loader is not None:
                val_results = self.validate(val_loader)
                val_loss = val_results['val_loss']
                self.val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    is_best = True
                    torch.save(self.model.state_dict(), 'best_model.pth')

                    if self.visualizer is not None:
                        self.plot_validation_comparison(val_results['val_batch_data'], epoch)

                log_msg = f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f} (Epoch {best_epoch})"
                self.logger.info(log_msg)
            else:
                log_msg = f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Best: {avg_train_loss:.4f}"
                self.logger.info(log_msg)

        if self.visualizer is not None:
            self.visualizer.plot_loss_curve(self.train_losses, self.val_losses, 'loss_curve.png')
            self.logger.info(f"Loss curve saved to {os.path.join(self.visualizer.save_dir, 'loss_curve.png')}")

        self.logger.info("Training completed!")
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch
        }


if __name__ == "__main__":
    from model import DyTR_LSTM
    from dataset import TimeSeriesDataset
    from visualizer import VisualUtils

    model = DyTR_LSTM()
    dataset = TimeSeriesDataset(['processed_trip.pkl'])
    visualizer = VisualUtils(state_names=['vlon', 'vlat', 'yaw', 'omega'], save_dir='./plots')

    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    trainer = Trainer(model, dataset.state_scaler, dataset.diff_scaler, visualizer=visualizer)

    trainer.train(train_loader, None, epochs=10, log_interval=1)
