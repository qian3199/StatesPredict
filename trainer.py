import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import logging
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, state_scaler, diff_scaler, lr=1e-4, weight_decay=1e-5, visualizer=None, log_dir='./logs', device=None, 
                 early_stopping=True, patience=20, min_delta=1e-8):
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
        
        # 时间记录相关变量
        self.epoch_times = []
        self.train_step_times = []
        self.val_times = []
        self.total_train_time = 0.0
        self.model_name = model.__class__.__name__  # 记录模型名称
        
        # 早停相关参数
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0

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
        base_mse = torch.mean((base_real - gt_real) ** 2).item()
        corr_mse = torch.mean((base_pred - gt_real) ** 2).item()

        self.logger.info(f"  Base RMSE: {base_error:.8f} | Net RMSE: {corr_error:.8f} | Base MSE: {base_mse:.10f} | Net MSE: {corr_mse:.10f} | Better: {corr_mse < base_mse}")

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
        self.logger.info(f"Model: {self.model_name}")
        
        # 记录总训练开始时间
        total_start_time = time.time()

        for epoch in range(epochs):
            # 记录epoch开始时间
            epoch_start_time = time.time()
            
            self.model.teacher_forcing_ratio = max(0.1, 0.5 - epoch * 0.004)

            train_losses = []
            train_step_start = time.time()
            for hist_state, hist_control, base_norm, cfg_tensor, gt_norm, diff_norm, _ in train_loader:
                losses = self.train_step(hist_state, hist_control, base_norm, cfg_tensor, gt_norm, diff_norm)
                train_losses.append(losses)
            train_step_time = time.time() - train_step_start
            self.train_step_times.append(train_step_time)

            avg_train_loss = np.mean([l['total_loss'] for l in train_losses])
            avg_cascade = np.mean([l['cascade_loss'] for l in train_losses])
            avg_mse = np.mean([l['mse_loss'] for l in train_losses])
            avg_r2 = np.mean([l['r2_score'] for l in train_losses])
            avg_base_mse = np.mean([l['base_mse'] for l in train_losses])
            avg_pred_mse = np.mean([l['pred_mse'] for l in train_losses])
            self.train_losses.append(avg_train_loss)

            is_best = False
            val_time = 0.0
            if val_loader is not None:
                val_start_time = time.time()
                val_results = self.validate(val_loader)
                val_time = time.time() - val_start_time
                self.val_times.append(val_time)
                
                val_loss = val_results['val_loss']
                val_mse = val_results['val_mse']
                val_r2 = val_results['val_r2']
                val_base_mse = val_results['val_base_mse']
                val_pred_mse = val_results['val_pred_mse']
                self.val_losses.append(val_loss)

                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    is_best = True
                    self.patience_counter = 0  # 重置耐心计数器
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    self.logger.info(f"  New best model saved! Base MSE: {val_base_mse:.8f} | Net MSE: {val_pred_mse:.8f} | Improvement: {(val_base_mse - val_pred_mse):.8f}")

                    if self.visualizer is not None:
                        self.plot_validation_comparison(val_results['val_batch_data'], epoch)
                else:
                    # 验证损失没有改善，增加耐心计数器
                    if self.early_stopping:
                        self.patience_counter += 1
                        self.logger.info(f"  Patience counter: {self.patience_counter}/{self.patience}")

            # 记录epoch结束时间
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            
            # 计算平均每batch时间
            avg_train_time_per_batch = train_step_time / len(train_loader)
            avg_val_time_per_batch = val_time / len(val_loader) if val_loader is not None else 0.0

            if val_loader is not None:
                log_msg = (f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                          f"MSE: {val_mse:.8f} | R2: {val_r2:.6f} | "
                          f"Base MSE: {val_base_mse:.8f} | Net MSE: {val_pred_mse:.8f} | "
                          f"Best: {best_val_loss:.6f} (Epoch {best_epoch}) | "
                          f"Time: {epoch_time:.2f}s (Train: {train_step_time:.2f}s, Val: {val_time:.2f}s)")
                self.logger.info(log_msg)
            else:
                log_msg = (f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f} | "
                          f"MSE: {avg_mse:.8f} | R2: {avg_r2:.6f} | "
                          f"Base MSE: {avg_base_mse:.8f} | Net MSE: {avg_pred_mse:.8f} | "
                          f"Time: {epoch_time:.2f}s")
                self.logger.info(log_msg)
            
            # 早停检查
            if self.early_stopping and val_loader is not None:
                if self.patience_counter >= self.patience:
                    self.logger.info(f"Early stopping triggered! No improvement in {self.patience} epochs.")
                    self.logger.info(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
                    break

        # 记录总训练结束时间
        total_train_time = time.time() - total_start_time
        self.total_train_time = total_train_time

        if val_loader is not None and self.visualizer is not None:
            self.plot_final_comparison(val_loader, best_epoch)

        if self.visualizer is not None:
            self.visualizer.plot_loss_curve(self.train_losses, self.val_losses, 'loss_curve.png')
            self.logger.info(f"Loss curve saved to {os.path.join(self.visualizer.save_dir, 'loss_curve.png')}")

        # 生成时间统计报告
        self.save_time_statistics(epochs, train_loader, val_loader)
        
        self.logger.info("Training completed!")
        self.logger.info(f"Total training time: {total_train_time:.2f}s ({total_train_time/60:.2f} minutes)")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'total_train_time': total_train_time,
            'avg_epoch_time': np.mean(self.epoch_times),
            'model_name': self.model_name
        }

    def plot_final_comparison(self, val_loader, best_epoch):
        self.logger.info("\n" + "="*80)
        self.logger.info("Generating final error comparison between Base and Net predictions...")
        
        self.model.eval()
        
        all_base_errors = []
        all_net_errors = []
        all_base_mse = []
        all_net_mse = []
        
        with torch.no_grad():
            for batch_idx, (hist_state, hist_control, base_norm, cfg_tensor, gt_norm, diff_norm, future_pos) in enumerate(val_loader):
                hist_state = hist_state.to(self.device)
                hist_control = hist_control.to(self.device)
                base_norm = base_norm.to(self.device)
                cfg_tensor = cfg_tensor.to(self.device)
                gt_norm = gt_norm.to(self.device)
                
                delta_norm, corrected_norm = self.model(hist_state, hist_control, base_norm, cfg_tensor)
                
                base_real = self.denormalize_state(base_norm)
                gt_real = self.denormalize_state(gt_norm)
                net_real = base_real + self.denormalize_diff(delta_norm)
                
                base_rmse = torch.sqrt(torch.mean((base_real - gt_real) ** 2, dim=[1, 2])).cpu().numpy()
                net_rmse = torch.sqrt(torch.mean((net_real - gt_real) ** 2, dim=[1, 2])).cpu().numpy()
                
                base_mse = torch.mean((base_real - gt_real) ** 2, dim=[1, 2]).cpu().numpy()
                net_mse = torch.mean((net_real - gt_real) ** 2, dim=[1, 2]).cpu().numpy()
                
                all_base_errors.extend(base_rmse.tolist())
                all_net_errors.extend(net_rmse.tolist())
                all_base_mse.extend(base_mse.tolist())
                all_net_mse.extend(net_mse.tolist())
        
        all_base_errors = np.array(all_base_errors)
        all_net_errors = np.array(all_net_errors)
        all_base_mse = np.array(all_base_mse)
        all_net_mse = np.array(all_net_mse)
        
        improvement = all_base_mse - all_net_mse
        improvement_ratio = (all_base_mse - all_net_mse) / (all_base_mse + 1e-8) * 100
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        ax1 = axes[0, 0]
        ax1.plot(all_base_errors, 'b-', alpha=0.5, label='Base RMSE', linewidth=0.5)
        ax1.plot(all_net_errors, 'r-', alpha=0.5, label='Net RMSE', linewidth=0.5)
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('RMSE')
        ax1.set_title('RMSE Comparison: Base vs Net (per sample)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        ax2.hist(all_base_errors, bins=50, alpha=0.5, label='Base RMSE', color='blue')
        ax2.hist(all_net_errors, bins=50, alpha=0.5, label='Net RMSE', color='red')
        ax2.set_xlabel('RMSE')
        ax2.set_ylabel('Frequency')
        ax2.set_title('RMSE Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        indices = np.arange(len(improvement))
        ax3.bar(indices, improvement, alpha=0.7, color='green')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('MSE Improvement (Base - Net)')
        ax3.set_title('MSE Improvement per Sample')
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        ax4.scatter(all_base_mse, all_net_mse, alpha=0.3, s=10)
        max_val = max(np.max(all_base_mse), np.max(all_net_mse))
        ax4.plot([0, max_val], [0, max_val], 'r--', label='y=x (No improvement)')
        ax4.set_xlabel('Base MSE')
        ax4.set_ylabel('Net MSE')
        ax4.set_title('Base MSE vs Net MSE')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.log_dir, 'final_error_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"\nFinal Error Comparison Summary:")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Base  - Mean RMSE: {np.mean(all_base_errors):.8f}, Mean MSE: {np.mean(all_base_mse):.8f}")
        self.logger.info(f"Net   - Mean RMSE: {np.mean(all_net_errors):.8f}, Mean MSE: {np.mean(all_net_mse):.8f}")
        self.logger.info(f"Improvement - Mean MSE reduction: {np.mean(improvement):.8f} ({np.mean(improvement_ratio):.2f}%)")
        self.logger.info(f"Best model from Epoch: {best_epoch}")
        self.logger.info(f"Comparison plot saved to: {save_path}")
        self.logger.info(f"{'='*80}\n")
        
        results_df = {
            'sample_idx': np.arange(len(all_base_errors)),
            'base_rmse': all_base_errors,
            'net_rmse': all_net_errors,
            'base_mse': all_base_mse,
            'net_mse': all_net_mse,
            'mse_improvement': improvement,
            'improvement_ratio_%': improvement_ratio
        }
        
        csv_path = os.path.join(self.log_dir, 'error_comparison_results.csv')
        try:
            import pandas as pd
            df = pd.DataFrame(results_df)
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Error comparison results saved to: {csv_path}")
        except ImportError:
            self.logger.warning("pandas not available, skipping CSV export")

    def save_time_statistics(self, epochs, train_loader, val_loader=None):
        """保存时间统计报告"""
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING TIME STATISTICS")
        self.logger.info("="*80)
        
        # 计算统计信息
        avg_epoch_time = np.mean(self.epoch_times)
        std_epoch_time = np.std(self.epoch_times)
        min_epoch_time = np.min(self.epoch_times)
        max_epoch_time = np.max(self.epoch_times)
        
        avg_train_step_time = np.mean(self.train_step_times)
        avg_val_time = np.mean(self.val_times) if self.val_times else 0.0
        
        num_train_batches = len(train_loader)
        num_val_batches = len(val_loader) if val_loader else 0
        
        avg_train_time_per_batch = avg_train_step_time / num_train_batches
        avg_val_time_per_batch = avg_val_time / num_val_batches if num_val_batches > 0 else 0.0
        
        # 输出到日志
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Total Epochs: {epochs}")
        self.logger.info(f"Total Training Time: {self.total_train_time:.2f}s ({self.total_train_time/60:.2f} minutes)")
        self.logger.info(f"\nEpoch Statistics:")
        self.logger.info(f"  Average Epoch Time: {avg_epoch_time:.2f}s")
        self.logger.info(f"  Std Epoch Time: {std_epoch_time:.2f}s")
        self.logger.info(f"  Min Epoch Time: {min_epoch_time:.2f}s (Epoch {np.argmin(self.epoch_times)})")
        self.logger.info(f"  Max Epoch Time: {max_epoch_time:.2f}s (Epoch {np.argmax(self.epoch_times)})")
        self.logger.info(f"\nTraining Step Statistics:")
        self.logger.info(f"  Average Train Step Time: {avg_train_step_time:.2f}s")
        self.logger.info(f"  Average Time per Batch: {avg_train_time_per_batch:.4f}s")
        self.logger.info(f"  Number of Train Batches: {num_train_batches}")
        if val_loader:
            self.logger.info(f"\nValidation Statistics:")
            self.logger.info(f"  Average Validation Time: {avg_val_time:.2f}s")
            self.logger.info(f"  Average Time per Batch: {avg_val_time_per_batch:.4f}s")
            self.logger.info(f"  Number of Val Batches: {num_val_batches}")
        
        self.logger.info("="*80 + "\n")
        
        # 保存到文件
        time_stats = {
            'model_name': self.model_name,
            'total_epochs': epochs,
            'total_train_time_s': self.total_train_time,
            'total_train_time_min': self.total_train_time / 60,
            'avg_epoch_time_s': avg_epoch_time,
            'std_epoch_time_s': std_epoch_time,
            'min_epoch_time_s': min_epoch_time,
            'max_epoch_time_s': max_epoch_time,
            'avg_train_step_time_s': avg_train_step_time,
            'avg_val_time_s': avg_val_time,
            'num_train_batches': num_train_batches,
            'num_val_batches': num_val_batches,
            'avg_train_time_per_batch_s': avg_train_time_per_batch,
            'avg_val_time_per_batch_s': avg_val_time_per_batch,
            'epoch_times': self.epoch_times,
            'train_step_times': self.train_step_times,
            'val_times': self.val_times
        }
        
        # 保存为JSON格式
        import json
        json_path = os.path.join(self.log_dir, 'time_statistics.json')
        with open(json_path, 'w') as f:
            json.dump(time_stats, f, indent=2)
        self.logger.info(f"Time statistics saved to: {json_path}")
        
        # 保存为CSV格式（epoch级别）
        try:
            import pandas as pd
            epoch_stats_df = {
                'epoch': np.arange(epochs),
                'epoch_time_s': self.epoch_times,
                'train_step_time_s': self.train_step_times,
                'val_time_s': self.val_times if self.val_times else [0] * epochs
            }
            df = pd.DataFrame(epoch_stats_df)
            csv_path = os.path.join(self.log_dir, 'epoch_time_statistics.csv')
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Epoch time statistics saved to: {csv_path}")
        except ImportError:
            self.logger.warning("pandas not available, skipping CSV export")
        
        # 生成时间曲线图
        self.plot_time_statistics()
    
    def plot_time_statistics(self):
        """绘制时间统计曲线图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = np.arange(len(self.epoch_times))
        
        # 子图1：每个epoch的总时间
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.epoch_times, 'b-', linewidth=2, marker='o', markersize=3)
        ax1.axhline(y=np.mean(self.epoch_times), color='r', linestyle='--', label=f'Mean: {np.mean(self.epoch_times):.2f}s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title(f'Epoch Time - {self.model_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2：训练步骤时间
        ax2 = axes[0, 1]
        ax2.plot(epochs, self.train_step_times, 'g-', linewidth=2, marker='s', markersize=3)
        ax2.axhline(y=np.mean(self.train_step_times), color='r', linestyle='--', label=f'Mean: {np.mean(self.train_step_times):.2f}s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Training Step Time per Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 子图3：验证时间
        ax3 = axes[1, 0]
        if self.val_times:
            ax3.plot(epochs, self.val_times, 'm-', linewidth=2, marker='^', markersize=3)
            ax3.axhline(y=np.mean(self.val_times), color='r', linestyle='--', label=f'Mean: {np.mean(self.val_times):.2f}s')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Validation Time per Epoch')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 子图4：时间分布对比
        ax4 = axes[1, 1]
        ax4.bar(['Train', 'Validation'], [np.sum(self.train_step_times), np.sum(self.val_times)], 
               color=['green', 'magenta'], alpha=0.7)
        ax4.set_ylabel('Total Time (seconds)')
        ax4.set_title(f'Total Time Distribution - {self.total_train_time:.2f}s')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.log_dir, 'time_statistics.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Time statistics plot saved to: {save_path}")


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
