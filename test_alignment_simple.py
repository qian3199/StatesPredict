import sys
sys.path.insert(0, '.')

from dataset import TimeSeriesDataset
import numpy as np

# 测试数据集
test_files = ['./data/left_turn_dataset/trip_0001.pkl']
dataset = TimeSeriesDataset(test_files, hist_len=15, pred_len=20, step=100, train=False)

print("=== 时间对齐测试 ===")
print(f"数据集大小: {len(dataset)}")

# 获取一个样本
idx = 0
hist_state, hist_control, base_pred, cfg_tensor, gt_pred, diff_pred, future_pos = dataset[idx]

print(f"\n样本 {idx} 的形状:")
print(f"  hist_state: {hist_state.shape}")
print(f"  base_pred: {base_pred.shape}")
print(f"  gt_pred: {gt_pred.shape}")

# 反归一化
base_real = dataset.base_scaler.inverse_transform(base_pred.numpy())
gt_real = dataset.state_scaler.inverse_transform(gt_pred.numpy())

print("\n=== 状态值对比 ===")
print("时间步 | 真实Vlon | 基线Vlon | 真实Yaw | 基线Yaw")
print("-" * 60)
for i in range(min(10, len(gt_real))):
    print(f"{i:6d} | {gt_real[i,0]:.4f} | {base_real[i,0]:.4f} | {gt_real[i,2]:.4f} | {base_real[i,2]:.4f}")

# 计算相关性
print("\n=== 相关性分析 ===")
for i, name in enumerate(['Vlon', 'Vlat', 'Yaw', 'Omega']):
    corr = np.corrcoef(gt_real[:, i], base_real[:, i])[0, 1]
    print(f"{name}: 相关性 = {corr:.4f}")

# 计算均方误差
mse = np.mean((gt_real - base_real) ** 2, axis=0)
print(f"\n基线预测 MSE: Vlon={mse[0]:.6f}, Vlat={mse[1]:.6f}, Yaw={mse[2]:.6f}, Omega={mse[3]:.6f}")

print("\n测试完成！")
