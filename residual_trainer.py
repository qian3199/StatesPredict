import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from log import logger


class ResidualTrainer:
    def __init__(self, model, optimizer, device, 
                 criterion=None,          # 新增：外部传入的损失函数
                 weights=None,           # 状态维度的权重
                 scheduler=None,
                 grad_clip_norm=1.0):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm
        
        # 优先级：外部传入的 criterion > 基于 weights 的自定义损失 > 默认 MSE
        if criterion is not None:
            self.criterion = criterion
        elif weights is not None:
            self.weights = weights.to(device)
            self.criterion = self._weighted_mse_loss
        else:
            self.criterion = torch.nn.MSELoss()
    
    def _weighted_mse_loss(self, pred, target):
        diff = pred - target
        return (diff ** 2 * self.weights.unsqueeze(0).unsqueeze(0)).mean()
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch in dataloader:
            # TimeSeriesDataset 返回: hist_state, hist_control, base_pred, cfg, gt_pred, diff_pred
            hist_state, hist_control, base_pred, cfg, gt_pred, diff_gt = batch
            
            hist_state = hist_state.to(self.device)
            hist_control = hist_control.to(self.device)
            base_pred = base_pred.to(self.device)
            cfg = cfg.to(self.device)
            diff_gt = diff_gt.to(self.device)
            
            # 前向传播
            residuals, predictions = self.model(hist_state, hist_control, base_pred, c=cfg)
            
            # 损失：预测残差 vs 真实残差
            loss = self.criterion(residuals, diff_gt)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            
            total_loss += loss.item() * hist_state.size(0)
            total_samples += hist_state.size(0)
        
        return total_loss / total_samples
    
    @torch.no_grad()
    def eval_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        for batch in dataloader:
            hist_state, hist_control, base_pred, cfg, gt_pred, diff_gt = batch
            hist_state = hist_state.to(self.device)
            hist_control = hist_control.to(self.device)
            base_pred = base_pred.to(self.device)
            cfg = cfg.to(self.device)
            diff_gt = diff_gt.to(self.device)
            
            residuals, predictions = self.model(hist_state, hist_control, base_pred, c=cfg)
            loss = self.criterion(residuals, diff_gt)
            
            total_loss += loss.item() * hist_state.size(0)
            total_samples += hist_state.size(0)
        
        return total_loss / total_samples
    
    def train(self, num_epochs, train_loader, val_loader, checkpoint_path,
              patience=30, delta_loss=1e-4, early_step=1):
        """
        训练主循环
        :param num_epochs: 最大训练轮数
        :param train_loader: 训练数据加载器
        :param val_loader: 验证数据加载器
        :param checkpoint_path: 模型保存路径（文件夹）
        :param patience: 早停耐心值
        :param delta_loss: 验证损失改善的最小阈值
        :param early_step: 每隔 early_step 个 epoch 打印一次详细日志（可选）
        :return: (train_losses, val_losses)
        """
        best_val_loss = float('inf')
        wait = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.eval_epoch(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 日志输出
            if epoch % early_step == 0 or epoch == 1:
                logger.info(f"Epoch {epoch:3d}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # 学习率调度（如果提供）
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 早停判断
            if val_loss < best_val_loss - delta_loss:
                best_val_loss = val_loss
                wait = 0
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, f"{checkpoint_path}/best_model.pth")
                logger.info(f"  保存最佳模型，验证损失: {best_val_loss:.6f}")
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"早停于 epoch {epoch}，最佳验证损失: {best_val_loss:.6f}")
                    break
        
        # 保存最终模型（可选）
        torch.save(self.model.state_dict(), f"{checkpoint_path}/last_model.pth")
        logger.info(f"训练结束，最佳验证损失: {best_val_loss:.6f}")
        
        return train_losses, val_losses