
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from generate_left_turn_data import generate_single_left_turn
import os


def test_data():
    # 生成几个测试数据
    print("生成测试数据...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i in range(3):
        data = generate_single_left_turn(i, dt=0.01, random_seed=i)
        if data is None:
            continue
        
        trip_id, (real_states, base_states, controls) = data
        
        # 绘制轨迹
        axes[i].plot(real_states[:, 0], real_states[:, 1], 'b-', label='Real (with noise)', linewidth=2)
        axes[i].plot(base_states[:, 0], base_states[:, 1], 'r--', label='Base (no noise)', linewidth=2)
        axes[i].set_xlabel('X (m)')
        axes[i].set_ylabel('Y (m)')
        axes[i].set_title(f'Trajectory (Trip {trip_id})')
        axes[i].legend()
        axes[i].grid(True)
        axes[i].axis('equal')
        
        # 保存到新的数据目录
        output_dir = './data/left_turn_dataset_smoothed'
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'trip_{trip_id:04d}.pkl'), 'wb') as f:
            pickle.dump([real_states, base_states, controls], f)
    
    plt.tight_layout()
    plt.savefig('test_smoothed_trips.png', dpi=150)
    print(f"已保存: test_smoothed_trips.png")
    plt.show()


if __name__ == '__main__':
    test_data()

