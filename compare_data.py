
import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob
import os


def load_data(data_dir, num_files=5):
    """加载并可视化几个样本"""
    pkl_files = sorted(glob.glob(os.path.join(data_dir, '*.pkl')))
    if not pkl_files:
        print(f"在 {data_dir} 中找不到文件")
        return []
    
    data_list = []
    for i, f in enumerate(pkl_files[:num_files]):
        with open(f, 'rb') as file:
            data = pickle.load(file)
        trip_id = data[0]
        real_states = data[1][0]
        base_states = data[1][1]
        controls = data[1][2]
        data_list.append((trip_id, real_states, base_states, controls))
    return data_list


def compare_data():
    # 加载两个数据集
    print("加载原始数据...")
    original_data = load_data('./data/left_turn_dataset', num_files=3)
    
    print("加载平滑数据...")
    smoothed_data = load_data('./data/left_turn_dataset_smoothed', num_files=3)
    
    if not original_data and not smoothed_data:
        print("没有找到数据！")
        return
    
    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 绘制原始数据
    for i, (trip_id, real_states, base_states, controls) in enumerate(original_data):
        axes[0, i].plot(real_states[:, 0], real_states[:, 1], 'b-', linewidth=2, label='Real (with noise)')
        axes[0, i].plot(base_states[:, 0], base_states[:, 1], 'r--', linewidth=2, label='Base (no noise)')
        axes[0, i].set_title(f'Original Data - {trip_id}', fontsize=14, fontweight='bold')
        axes[0, i].set_xlabel('X (m)')
        axes[0, i].set_ylabel('Y (m)')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].axis('equal')
    
    # 绘制平滑数据
    for i, (trip_id, real_states, base_states, controls) in enumerate(smoothed_data):
        axes[1, i].plot(real_states[:, 0], real_states[:, 1], 'b-', linewidth=2, label='Real (with noise)')
        axes[1, i].plot(base_states[:, 0], base_states[:, 1], 'r--', linewidth=2, label='Base (no noise)')
        axes[1, i].set_title(f'Smoothed Data - {trip_id}', fontsize=14, fontweight='bold')
        axes[1, i].set_xlabel('X (m)')
        axes[1, i].set_ylabel('Y (m)')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].axis('equal')
    
    plt.tight_layout(pad=3.0)
    plt.savefig('data_comparison.png', dpi=150, bbox_inches='tight')
    print(f"已保存对比图: data_comparison.png")
    plt.show()
    
    # 绘制状态量对比
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 12))
    
    # 原始数据状态量
    trip_id, real_states, base_states, controls = original_data[0]
    time = np.arange(len(real_states)) * 0.01
    state_names = ['X', 'Y', 'Vlon', 'Vlat', 'Yaw', 'Omega', 'AccLon', 'AccLat', 'AccYaw']
    
    for i in range(4):
        axes2[0, i].plot(time, real_states[:, i+2], 'b-', linewidth=2, label='Real')
        axes2[0, i].plot(time, base_states[:, i+2], 'r--', linewidth=2, label='Base')
        axes2[0, i].set_title(f'Original - {state_names[i+2]}', fontsize=12, fontweight='bold')
        axes2[0, i].set_xlabel('Time (s)')
        axes2[0, i].set_ylabel(state_names[i+2])
        axes2[0, i].legend()
        axes2[0, i].grid(True, alpha=0.3)
    
    # 平滑数据状态量
    trip_id, real_states, base_states, controls = smoothed_data[0]
    time = np.arange(len(real_states)) * 0.01
    
    for i in range(4):
        axes2[1, i].plot(time, real_states[:, i+2], 'b-', linewidth=2, label='Real')
        axes2[1, i].plot(time, base_states[:, i+2], 'r--', linewidth=2, label='Base')
        axes2[1, i].set_title(f'Smoothed - {state_names[i+2]}', fontsize=12, fontweight='bold')
        axes2[1, i].set_xlabel('Time (s)')
        axes2[1, i].set_ylabel(state_names[i+2])
        axes2[1, i].legend()
        axes2[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('state_comparison.png', dpi=150, bbox_inches='tight')
    print(f"已保存状态量对比图: state_comparison.png")
    plt.show()


if __name__ == '__main__':
    compare_data()

