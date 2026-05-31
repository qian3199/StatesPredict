"""
简单的推理测试脚本 - 可视化训练后的模型效果
"""
import torch
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os

from model import DyTR_LSTM
from dataset import TimeSeriesDataset

def run_inference_test():
    print("=" * 60)
    print("推理测试 - 使用训练好的模型")
    print("=" * 60)
    
    # 加载训练数据的一个样本
    data_dir = './data/left_turn_dataset'
    pkl_files = sorted(glob.glob(os.path.join(data_dir, '*.pkl')))
    
    # 加载一个验证样本
    val_file = pkl_files[40]  # 使用第一个验证样本
    print(f"\n加载测试数据: {val_file}")
    
    with open(val_file, 'rb') as f:
        trip_id, [real_states, base_states, controls] = pickle.load(f)
    
    print(f"轨迹ID: {trip_id}")
    print(f"数据形状: real_states={real_states.shape}, base_states={base_states.shape}")
    
    # 创建数据集获取scaler
    train_files = pkl_files[:40]
    train_dataset = TimeSeriesDataset(
        train_files, 
        hist_len=15, 
        pred_len=1, 
        step=1, 
        train=True
    )
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    model = DyTR_LSTM().to(device)
    model.load_state_dict(torch.load('left_turn_model.pth', map_location=device))
    model.eval()
    
    print("模型加载成功!")
    
    # 状态索引: [2, 3, 4, 5] = [vlon, vlat, yaw, omega]
    state_indices = [2, 3, 4, 5]
    
    # 准备测试数据
    hist_len = 15
    hist_states = real_states[:hist_len, state_indices]  # 只取4个状态
    hist_controls = controls[:hist_len]  # 控制是 [acc, steering]
    base_state = base_states[hist_len:hist_len+50][:, state_indices]  # 预测50步
    real_future = real_states[hist_len:hist_len+50][:, state_indices]
    
    # 转换为tensor
    hist_state_t = torch.FloatTensor(hist_states).unsqueeze(0).to(device)
    hist_control_t = torch.FloatTensor(hist_controls).unsqueeze(0).to(device)
    base_state_t = torch.FloatTensor(base_state).unsqueeze(0).to(device)
    cfg_tensor = torch.FloatTensor([15, 1, 1]).unsqueeze(0).to(device)
    
    # 归一化
    state_mean = torch.FloatTensor(train_dataset.state_scaler.mean_).to(device)
    state_std = torch.FloatTensor(train_dataset.state_scaler.scale_).to(device)
    control_mean = torch.FloatTensor(train_dataset.control_scaler.mean_).to(device)
    control_std = torch.FloatTensor(train_dataset.control_scaler.scale_).to(device)
    
    hist_state_norm = (hist_state_t - state_mean) / (state_std + 1e-8)
    hist_control_norm = (hist_control_t - control_mean) / (control_std + 1e-8)
    base_state_norm = (base_state_t - state_mean) / (state_std + 1e-8)
    
    # 推理
    with torch.no_grad():
        delta_norm, corrected_norm = model(hist_state_norm, hist_control_norm, base_state_norm, cfg_tensor)
    
    # 反归一化
    corrected_real = corrected_norm * (state_std + 1e-8) + state_mean
    corrected_real = corrected_real.squeeze(0).cpu().numpy()
    
    # 计算轨迹
    def integrate_trajectory(states, dt=0.01):
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
    
    # 从初始点开始积分
    start_xy = real_states[hist_len-1, :2]
    
    real_traj = integrate_trajectory(real_future, dt=0.01)
    base_traj = integrate_trajectory(base_state, dt=0.01)
    pred_traj = integrate_trajectory(corrected_real, dt=0.01)
    
    # 平移到起点
    real_traj += start_xy
    base_traj += start_xy
    pred_traj += start_xy
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 轨迹对比
    ax1 = axes[0, 0]
    ax1.plot(real_traj[:, 0], real_traj[:, 1], 'g-', linewidth=2, label='Real (Ground Truth)')
    ax1.plot(base_traj[:, 0], base_traj[:, 1], 'b--', linewidth=2, label='Base (Physics)')
    ax1.plot(pred_traj[:, 0], pred_traj[:, 1], 'r-', linewidth=2, label='Hybrid Prediction')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('轨迹对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. 速度对比
    ax2 = axes[0, 1]
    time = np.arange(50) * 0.01
    ax2.plot(time, real_future[:, 0], 'g-', linewidth=2, label='Real vlon')
    ax2.plot(time, base_state[:, 0], 'b--', linewidth=2, label='Base vlon')
    ax2.plot(time, corrected_real[:, 0], 'r-', linewidth=2, label='Pred vlon')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('纵向速度对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 横向速度对比
    ax3 = axes[1, 0]
    ax3.plot(time, real_future[:, 1], 'g-', linewidth=2, label='Real vlat')
    ax3.plot(time, base_state[:, 1], 'b--', linewidth=2, label='Base vlat')
    ax3.plot(time, corrected_real[:, 1], 'r-', linewidth=2, label='Pred vlat')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('横向速度对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 偏航角对比
    ax4 = axes[1, 1]
    ax4.plot(time, real_future[:, 2], 'g-', linewidth=2, label='Real yaw')
    ax4.plot(time, base_state[:, 2], 'b--', linewidth=2, label='Base yaw')
    ax4.plot(time, corrected_real[:, 2], 'r-', linewidth=2, label='Pred yaw')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Yaw (rad)')
    ax4.set_title('偏航角对比')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = './inference_test_result.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n推理结果已保存到: {save_path}")
    
    # 计算误差
    base_error = np.sqrt((base_traj[:, 0] - real_traj[:, 0])**2 + (base_traj[:, 1] - real_traj[:, 1])**2)
    pred_error = np.sqrt((pred_traj[:, 0] - real_traj[:, 0])**2 + (pred_traj[:, 1] - real_traj[:, 1])**2)
    
    print(f"\n轨迹误差统计:")
    print(f"  Base平均误差: {np.mean(base_error):.4f} m")
    print(f"  Pred平均误差: {np.mean(pred_error):.4f} m")
    print(f"  误差改善: {((np.mean(base_error) - np.mean(pred_error)) / np.mean(base_error) * 100):.2f}%")
    
    plt.show()

if __name__ == "__main__":
    run_inference_test()
