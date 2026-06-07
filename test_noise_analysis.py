
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


def analyze_noise():
    # 加载原始数据
    original_file = './data/left_turn_dataset/trip_0001.pkl'
    smooth_file = './data/left_turn_dataset_smoothed/trip_0001.pkl'
    
    with open(original_file, 'rb') as f:
        original_data = pickle.load(f)
    
    with open(smooth_file, 'rb') as f:
        smooth_data = pickle.load(f)
    
    original_real = original_data[1][0]
    smooth_real = smooth_data[1][0]
    
    dt = 0.01
    time = np.arange(len(original_real)) * dt
    
    # 计算状态量的波动（标准差）
    print("状态量波动分析：")
    state_names = ['X', 'Y', 'Vlon', 'Vlat', 'Yaw', 'Omega', 'AccLon', 'AccLat', 'AccYaw']
    for i in range(len(state_names)):
        orig_std = np.std(np.diff(original_real[:, i]))
        smoo_std = np.std(np.diff(smooth_real[:, i]))
        print(f"{state_names[i]:<10}: 原始数据波动={orig_std:.6f}, 平滑数据波动={smoo_std:.6f}, 差异={orig_std/smoo_std:.2f}x")
    
    # 绘制对比图
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
    
    for i in range(min(9, original_real.shape[1])):
        ax = axes[i]
        ax.plot(time, original_real[:, i], 'b-', label='Original', alpha=0.7)
        ax.plot(time, smooth_real[:, i], 'r--', label='Smoothed', alpha=0.7)
        ax.set_title(f'{state_names[i]}', fontsize=12)
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('noise_analysis.png', dpi=150)
    print("\n噪声分析图已保存: noise_analysis.png")
    plt.show()


if __name__ == '__main__':
    analyze_noise()

