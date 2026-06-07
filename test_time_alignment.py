import sys
sys.path.insert(0, '.')

from dataset import TimeSeriesDataset
import matplotlib.pyplot as plt
import numpy as np

# 测试数据集
test_files = ['./data/left_turn_dataset/trip_0001.pkl']
dataset = TimeSeriesDataset(test_files, hist_len=15, pred_len=20, step=100, train=False)

# 获取一个样本
idx = 0
hist_state, hist_control, base_pred, cfg_tensor, gt_pred, diff_pred, future_pos = dataset[idx]

print(f"样本 {idx} 的形状:")
print(f"  hist_state: {hist_state.shape}")
print(f"  base_pred: {base_pred.shape}")
print(f"  gt_pred: {gt_pred.shape}")

# 反归一化
base_real = dataset.base_scaler.inverse_transform(base_pred.numpy())
gt_real = dataset.state_scaler.inverse_transform(gt_pred.numpy())

# 绘制对比图
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
state_names = ['Vlon', 'Vlat', 'Yaw', 'Omega']

for i in range(4):
    ax = axes[i//2, i%2]
    ax.plot(gt_real[:, i], label='Ground Truth', color='green', linewidth=2)
    ax.plot(base_real[:, i], label='Physics Baseline', color='blue', linewidth=2, linestyle='--')
    ax.set_title(state_names[i], fontsize=14)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Time Alignment Test: Ground Truth vs Physics Baseline', fontsize=16)
plt.tight_layout()
plt.savefig('time_alignment_test.png', dpi=150)
print("\n时间对齐测试图已保存: time_alignment_test.png")
plt.show()

# 计算时间偏移
print("\n时间对齐分析:")
for i in range(4):
    # 计算基线和真实值之间的延迟
    cross_corr = np.correlate(gt_real[:, i], base_real[:, i], mode='full')
    delay = np.argmax(cross_corr) - (len(gt_real) - 1)
    print(f"{state_names[i]}: 检测到的时间延迟 = {delay} 步")
