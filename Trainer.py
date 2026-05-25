import os
import torch
import traceback
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from log import logger

class StateIndex:
    VLON = 0
    VLAT = 1
    YAW = 2
    VYAW = 3

class Trainer:
    def __init__(
        self, model, opt, device, weights, batch_size, normalize,
        dt=0.1, ar_steps=30, physics_weights=None, 
        warm_up_step=5,                             
        initial_tf_ratio=0.5,                     
        tf_decay_epoch=5
    ):
        self.model = model
        self.optimizer = opt
        self.device = device
        self.weights = weights.to(device)
        self.best_val_loss = float('inf')
        self.normalize = normalize
        self.batch_size = batch_size
        self.patience_counter = 0
        self.dt = dt
        self.ar_steps = ar_steps
        self.warm_up_step = warm_up_step

        self.initial_tf_ratio = initial_tf_ratio
        self.current_tf_ratio = initial_tf_ratio
        self.tf_decay_epoch = tf_decay_epoch

        if physics_weights is None:
            self.physics_weights = [1.0, 1.0, 0.5]
        else:
            self.physics_weights = physics_weights

        self.model = self.model.to(self.device)

    def _move_to_device(self, batch):
        hist_state, hist_control, base_state, cfg_tensor, gt_pred, diff_pred = batch
        return (hist_state.to(self.device), hist_control.to(self.device),
                base_state.to(self.device), cfg_tensor.to(self.device),
                gt_pred.to(self.device), diff_pred.to(self.device))

    def _prepare_scaler_tensors(self, scalers):
        if not self.normalize:
            return None
        device = self.device
        return {
            'base_mean': torch.tensor(scalers['base'].mean_).float().to(device),
            'base_std':  torch.tensor(scalers['base'].scale_).float().to(device),
            'diff_mean': torch.tensor(scalers['diff'].mean_).float().to(device),
            'diff_std':  torch.tensor(scalers['diff'].scale_).float().to(device),
            'state_mean': torch.tensor(scalers['state'].mean_).float().to(device),
            'state_std':  torch.tensor(scalers['state'].scale_).float().to(device),
        }

    def _apply_partial_correction(self, pred_raw, fallback_raw):
        mask = (self.weights > 0).float().view(1, 1, -1)
        final_state = mask * pred_raw + (1 - mask) * fallback_raw
        return final_state

    @staticmethod
    def integrate_trajectory_ar(state, dt):
        B, T, _ = state.shape
        device = state.device
        if T <= 1:
            return torch.zeros(B, T, device=device), torch.zeros(B, T, device=device)

        vlon = state[:, :, StateIndex.VLON]
        vlat = state[:, :, StateIndex.VLAT]
        yaw  = state[:, :, StateIndex.YAW]

        vx = vlon[:, :-1] * torch.cos(yaw[:, :-1]) - vlat[:, :-1] * torch.sin(yaw[:, :-1])
        vy = vlon[:, :-1] * torch.sin(yaw[:, :-1]) + vlat[:, :-1] * torch.cos(yaw[:, :-1])

        dx, dy = vx * dt, vy * dt
        init_x = torch.zeros(B, 1, device=device)
        init_y = torch.zeros(B, 1, device=device)

        x = torch.cat([init_x, torch.cumsum(dx, dim=1)], dim=1)
        y = torch.cat([init_y, torch.cumsum(dy, dim=1)], dim=1)
        return x, y

    def compute_cascade_loss(self, corrected_state, gt_raw):
        if torch.isnan(corrected_state).any() or torch.isinf(corrected_state).any():
            logger.error("NaN or Inf detected in corrected_state!")
            return torch.tensor(0.0, requires_grad=True).to(corrected_state.device), 0.0, 0.0
        if torch.isnan(gt_raw).any() or torch.isinf(gt_raw).any():
            logger.error("NaN or Inf detected in gt_raw!")
            return torch.tensor(0.0, requires_grad=True).to(corrected_state.device), 0.0, 0.0

        w = min(self.warm_up_step, corrected_state.shape[1] - 1)
        if w >= corrected_state.shape[1]:
            dummy_loss = torch.tensor(0.0, requires_grad=True).to(corrected_state.device)
            return dummy_loss, 0.0, 0.0

        pred_slice = corrected_state[:, w:]
        gt_slice   = gt_raw[:, w:]
        T = pred_slice.shape[1]

        if T <= 1:
            dummy_loss = torch.tensor(0.0, requires_grad=True).to(corrected_state.device)
            return dummy_loss, 0.0, 0.0

        max_weight = 3.0
        step_weights = torch.linspace(1.0, max_weight, T, device=corrected_state.device)

        pred_vlat = pred_slice[:, :, StateIndex.VLAT]
        gt_vlat   = gt_slice[:, :, StateIndex.VLAT]
        per_step_vlat_loss = torch.nn.functional.smooth_l1_loss(pred_vlat, gt_vlat, reduction='none')
        weighted_vlat_loss = per_step_vlat_loss * step_weights.unsqueeze(0)
        vlat_loss = weighted_vlat_loss.sum() / step_weights.sum()

        curr_pred = pred_slice[:, 1:, :]  # [B, T-1, dim]
        prev_pred = pred_slice[:, :-1, :] # [B, T-1, dim]
        curr_gt   = gt_slice[:, 1:, :]    # [B, T-1, dim]
        prev_gt   = gt_slice[:, :-1, :]   # [B, T-1, dim]

        gt_yaw_t = prev_gt[:, :, StateIndex.YAW]
        n_x = -torch.sin(gt_yaw_t)
        n_y = torch.cos(gt_yaw_t)

        vx_pred = prev_pred[:, :, StateIndex.VLON] * torch.cos(prev_pred[:, :, StateIndex.YAW]) - \
                  prev_pred[:, :, StateIndex.VLAT] * torch.sin(prev_pred[:, :, StateIndex.YAW])
        vy_pred = prev_pred[:, :, StateIndex.VLON] * torch.sin(prev_pred[:, :, StateIndex.YAW]) + \
                  prev_pred[:, :, StateIndex.VLAT] * torch.cos(prev_pred[:, :, StateIndex.YAW])
        dx_pred = vx_pred * self.dt
        dy_pred = vy_pred * self.dt

        vx_gt = prev_gt[:, :, StateIndex.VLON] * torch.cos(prev_gt[:, :, StateIndex.YAW]) - \
                prev_gt[:, :, StateIndex.VLAT] * torch.sin(prev_gt[:, :, StateIndex.YAW])
        vy_gt = prev_gt[:, :, StateIndex.VLON] * torch.sin(prev_gt[:, :, StateIndex.YAW]) + \
                prev_gt[:, :, StateIndex.VLAT] * torch.cos(prev_gt[:, :, StateIndex.YAW])
        dx_gt = vx_gt * self.dt
        dy_gt = vy_gt * self.dt

        err_x = dx_pred - dx_gt
        err_y = dy_pred - dy_gt
        lateral_error_step = err_x * n_x + err_y * n_y
        
        abs_lateral_error_step = torch.abs(lateral_error_step)
        
        step_weights_traj = step_weights[1:].unsqueeze(0) 
        
        weighted_lateral_error_step = abs_lateral_error_step * step_weights_traj
        traj_loss = weighted_lateral_error_step.sum() / step_weights_traj.sum()

        total_loss = (self.physics_weights[0] * vlat_loss + 
                      self.physics_weights[1] * traj_loss)
        return total_loss, vlat_loss.item(), traj_loss.item()


    def _check_gradient_explosion(self, threshold=5.0):
        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=threshold)
        if total_grad_norm > threshold:
            logger.debug(f"Gradient clipped! Norm: {total_grad_norm:.4f} (Threshold: {threshold})")

    def train_epoch(self, data_loader, scalers=None):
        self.model.train()
        train_epoch_loss, vlat_losses, traj_losses = [], [], []
        scaler_tensors = self._prepare_scaler_tensors(scalers) if self.normalize else None

        for batch in data_loader:
            hist_state, hist_control, base_state, cfg_tensor, gt_pred, diff_pred = self._move_to_device(batch)
            self.optimizer.zero_grad()

            delta_state_norm, corrected_state_norm = self.model(
                s_hist=hist_state, u_hist=hist_control, s_next=gt_pred, c=cfg_tensor, 
                base_state=base_state, return_corrected=True
            )
            
            if self.normalize:
                corrected_raw = corrected_state_norm * scaler_tensors['state_std'] + scaler_tensors['state_mean']
                delta_raw = delta_state_norm * scaler_tensors['diff_std'] + scaler_tensors['diff_mean']  # 用 diff 反归一化
                base_raw = base_state * scaler_tensors['state_std'] + scaler_tensors['state_mean']      # Base 用 state 反归一化
                gt_raw = gt_pred * scaler_tensors['state_std'] + scaler_tensors['state_mean']        # GT 用 state 反归一化
            else:
                corrected_raw = corrected_state_norm
                delta_raw = delta_state_norm
                base_raw = base_state
                gt_raw = gt_pred

            # final_pred_raw = base_raw + delta_raw
            final_pred_raw = self._apply_partial_correction(corrected_raw, base_raw)

            # final_pred_raw = self._apply_partial_correction(final_pred_raw, base_raw)

            cascade_loss, vlat_l, traj_l = self.compute_cascade_loss(final_pred_raw, gt_raw)                
            
            diff_direct_loss = torch.nn.functional.mse_loss(delta_state_norm, diff_pred)
            
            total_loss = cascade_loss + 0.1 * diff_direct_loss

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.error(f"NaN or Inf loss detected! Skipping batch.")
                continue

            total_loss.backward()
            self._check_gradient_explosion(threshold=5.0)
            self.optimizer.step()

            train_epoch_loss.append(total_loss.item())
            vlat_losses.append(vlat_l)
            traj_losses.append(traj_l)

        avg = lambda lst: sum(lst) / len(lst) if lst else 0
        return avg(train_epoch_loss), {'vlat_loss': avg(vlat_losses), 'traj_loss': avg(traj_losses), 'diff_loss': avg([diff_direct_loss])}
    
    def val_epoch(self, data_loader, scalers=None):
        self.model.eval()
        val_epoch_loss, vlat_losses, traj_losses = [], [], [] 
        all_model_states, all_base_states, all_gts = [], [], []
        scaler_tensors = self._prepare_scaler_tensors(scalers) if self.normalize else None

        with torch.no_grad():
            for batch in data_loader:
        
                hist_state, hist_control, base_state, cfg_tensor, gt_pred, diff_pred = self._move_to_device(batch)
                                  
                delta_state_norm, corrected_state_norm = self.model(
                    s_hist=hist_state,         
                    u_hist=hist_control,      
                    s_next=gt_pred,                      
                    c=cfg_tensor,                    
                    base_state=base_state, 
                    return_corrected=True
                )

                if self.normalize:
                    corrected_raw = corrected_state_norm * scaler_tensors['state_std'] + scaler_tensors['state_mean']
                    delta_raw = delta_state_norm * scaler_tensors['diff_std'] + scaler_tensors['diff_mean']
                    base_raw = base_state * scaler_tensors['state_std'] + scaler_tensors['state_mean']
                    gt_raw = gt_pred * scaler_tensors['state_std'] + scaler_tensors['state_mean']
                else:
                    corrected_raw = corrected_state_norm
                    delta_raw = delta_state_norm
                    base_raw = base_state
                    gt_raw = gt_pred

                # final_pred_raw = base_raw + delta_raw
                final_pred_raw = self._apply_partial_correction(corrected_raw, base_raw)

                # final_pred_raw = self._apply_partial_correction(final_pred_raw, base_raw)
                
                cascade_loss, vlat_l, traj_l = self.compute_cascade_loss(final_pred_raw, gt_raw)
                diff_direct_loss = torch.nn.functional.mse_loss(delta_state_norm, diff_pred)
                
                val_loss = cascade_loss + 0.1 * diff_direct_loss
                val_epoch_loss.append(val_loss.item())
                
                vlat_losses.append(vlat_l)
                traj_losses.append(traj_l)
                
                all_model_states.append(final_pred_raw.cpu())
                all_base_states.append(base_raw.cpu())
                all_gts.append(gt_raw.cpu())

        avg_val_loss = sum(val_epoch_loss) / len(val_epoch_loss) if val_epoch_loss else 0

        all_model_states = torch.cat(all_model_states, dim=0)
        all_base_states = torch.cat(all_base_states, dim=0)
        all_gts = torch.cat(all_gts, dim=0)
        base_error = all_base_states - all_gts
        model_error = all_model_states - all_gts
        
        avg = lambda lst: sum(lst) / len(lst) if lst else 0
        return avg_val_loss, all_model_states, all_base_states, all_gts, base_error, model_error, {'vlat_loss': avg(vlat_losses), 'traj_loss': avg(traj_losses), 'diff_loss': avg([diff_direct_loss])}


    def save_checkpoint(self, checkpoint_path, epoch, train_epoch_loss, val_epoch_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_epoch_loss': train_epoch_loss,
            'val_epoch_loss': val_epoch_loss
        }
        filename = 'best_dytr_model.pth' if is_best else f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, os.path.join(checkpoint_path, filename))
        logger.info(f"Saved model at epoch {epoch+1}")
        if is_best:
            logger.info(f"Best model saved with val loss {val_epoch_loss:.4f}")

    def train(self, num_epochs, patience, delta_loss, checkpoint_path,
            early_step, visualizer, train_dataset, val_loader, scalers=None):
        from torch.utils.data import DataLoader

        train_epoch_loss_list = []
        val_epoch_loss_list = []
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            logger.info(" ")
            logger.info(f"=== Training epoch {epoch+1}/{num_epochs} ===")

            self.current_tf_ratio = max(0.0, self.initial_tf_ratio * (0.95 ** epoch))
            logger.info(f"[AR Status] Teacher Forcing Ratio: {self.current_tf_ratio:.2f}")


            try:
                train_loss, train_detail = self.train_epoch(train_loader, scalers)
                train_epoch_loss_list.append(train_loss)

                logger.info(
                    f"[Train] epoch {epoch+1}/{num_epochs}, "
                    f"total_loss={train_loss:.4f}, "
                    f"vlat_mse={train_detail['vlat_loss']:.4f}, "
                    f"traj_ade={train_detail['traj_loss']:.4f}"
                )

                if (epoch + 1) % early_step == 0:
                    logger.info("Running validation (100% Closed-Loop)...")
                    try:
                        val_results = self.val_epoch(val_loader, scalers)
                        val_epoch_loss, val_model_state, val_base_state, val_gt, base_error, model_error, val_detail = val_results
                        val_epoch_loss_list.append(val_epoch_loss)
                        
                        logger.info(
                            f"★ [Validation] epoch {epoch+1}/{num_epochs}, "
                            f"Val Total Loss = {val_epoch_loss:.4f},"
                            f"vlat_mse={val_detail['vlat_loss']:.4f}, "
                            f"traj_ade={val_detail['traj_loss']:.4f}"
                        )

                        self.save_checkpoint(checkpoint_path, epoch, train_loss, val_epoch_loss, is_best=False)

                        if val_epoch_loss < self.best_val_loss - delta_loss:
                            self.best_val_loss = val_epoch_loss
                            self.patience_counter = 0
                            self.save_checkpoint(checkpoint_path, epoch, train_loss, val_epoch_loss, is_best=True)
                            logger.info(f"New best model saved! (Val Loss decreased to {val_epoch_loss:.4f})")
                        else:
                            self.patience_counter += 1
                            logger.info(f"Early stopping counter: {self.patience_counter}/{patience}")
                            if self.patience_counter >= patience:
                                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                                break

                        if visualizer is not None:
                            visualizer.visual_res_function(
                                epoch, val_gt, val_base_state, val_model_state,
                                base_error, model_error,
                                label=f"val_epoch_{epoch+1}", save_results=True
                            )
                            visualizer.visual_trajectory_function(
                                gt=val_gt, base_state=val_base_state, model_state=val_model_state,
                                dt=self.dt, label=f"traj_epoch_{epoch+1}", save_results=True, max_samples=200
                            )

                    except Exception as e:
                        logger.error(f"Error in validation epoch {epoch+1}: {e}")
                        traceback.print_exc()
                        continue
                else:
                    # ★ 修复6：补全 else，保证画图不报错
                    if val_epoch_loss_list:
                        val_epoch_loss_list.append(val_epoch_loss_list[-1])
                    else:
                        val_epoch_loss_list.append(float('inf'))

            except Exception as e:
                logger.error(f"Error in training epoch {epoch+1}: {e}")
                traceback.print_exc()
                if train_epoch_loss_list: train_epoch_loss_list.append(train_epoch_loss_list[-1])
                else: train_epoch_loss_list.append(float('inf'))
                if val_epoch_loss_list: val_epoch_loss_list.append(val_epoch_loss_list[-1])
                else: val_epoch_loss_list.append(float('inf'))
                continue

        return train_epoch_loss_list, val_epoch_loss_list