"""
生成最终的轨迹对比图 - 简化版
只展示Real vs Base的数据对比
"""
import pickle
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    # 加载数据
    data_dir = './data/left_turn_dataset'
    pkl_files = sorted(glob.glob(f'{data_dir}/*.pkl'))
    
    print(f"Found {len(pkl_files)} files")
    
    # 处理所有500条轨迹
    all_real = []
    all_base = []
    
    for idx, f in enumerate(pkl_files):
        with open(f, 'rb') as fp:
            name, [real, base, ctrl] = pickle.load(fp)
        
        # 平移到原点
        start_x, start_y = real[0, 0], real[0, 1]
        real_xy = real[:, 0:2] - np.array([start_x, start_y])
        base_xy = base[:, 0:2] - np.array([start_x, start_y])
        
        all_real.append(real_xy)
        all_base.append(base_xy)
        
        if idx % 100 == 0:
            print(f"  Loaded {idx}/{len(pkl_files)}")
    
    print("Done loading! Plotting...")
    
    # 画所有500条的real vs base
    fig, ax = plt.subplots(figsize=(16, 14))
    
    for i in range(len(pkl_files)):
        is_bold = i % 50 == 0
        alpha = 0.7 if is_bold else 0.2
        lw = 1.5 if is_bold else 0.5
        
        if i == 0:
            ax.plot(all_real[i][:, 0], all_real[i][:, 1], 'g-', alpha=alpha, linewidth=lw, label='Real (Ground Truth)')
            ax.plot(all_base[i][:, 0], all_base[i][:, 1], 'b--', alpha=alpha, linewidth=lw, label='Base (Physics Model)')
        else:
            ax.plot(all_real[i][:, 0], all_real[i][:, 1], 'g-', alpha=alpha, linewidth=lw)
            ax.plot(all_base[i][:, 0], all_base[i][:, 1], 'b--', alpha=alpha, linewidth=lw)
    
    ax.plot(0, 0, 'ko', markersize=15, label='Start (0,0)', zorder=10)
    ax.set_xlabel('X Position (m)', fontsize=14)
    ax.set_ylabel('Y Position (m)', fontsize=14)
    ax.set_title('All 500 Left Turn Trajectories\nGreen=Real, Blue dashed=Physics Model', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('images/all_500_trajectories_final.png', dpi=150, bbox_inches='tight')
    print("Saved all_500_trajectories_final.png")
    
    # 也画前50条的详细对比
    fig2, ax2 = plt.subplots(figsize=(16, 14))
    
    for i in range(50):
        is_bold = i % 5 == 0
        alpha = 0.9 if is_bold else 0.3
        lw = 2.0 if is_bold else 0.7
        
        if i == 0:
            ax2.plot(all_real[i][:, 0], all_real[i][:, 1], 'g-', alpha=alpha, linewidth=lw, label='Real (Ground Truth)')
            ax2.plot(all_base[i][:, 0], all_base[i][:, 1], 'b--', alpha=alpha, linewidth=lw, label='Base (Physics Model)')
        else:
            ax2.plot(all_real[i][:, 0], all_real[i][:, 1], 'g-', alpha=alpha, linewidth=lw)
            ax2.plot(all_base[i][:, 0], all_base[i][:, 1], 'b--', alpha=alpha, linewidth=lw)
    
    ax2.plot(0, 0, 'ko', markersize=15, label='Start (0,0)', zorder=10)
    ax2.set_xlabel('X Position (m)', fontsize=14)
    ax2.set_ylabel('Y Position (m)', fontsize=14)
    ax2.set_title('First 50 Left Turn Trajectories Detail\nGreen=Real, Blue dashed=Physics Model', fontsize=14)
    ax2.legend(fontsize=12, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('images/first_50_trajectories_final.png', dpi=150, bbox_inches='tight')
    print("Saved first_50_trajectories_final.png")
    
    # 计算并显示统计信息
    print("\n=== 轨迹误差统计 ===")
    errors = []
    for real_xy, base_xy in zip(all_real, all_base):
        # 计算每条轨迹的终点误差
        end_error = np.sqrt((real_xy[-1, 0] - base_xy[-1, 0])**2 + (real_xy[-1, 1] - base_xy[-1, 1])**2)
        errors.append(end_error)
    
    errors = np.array(errors)
    print(f"终点误差 - Mean: {errors.mean():.4f}m, Std: {errors.std():.4f}m")
    print(f"终点误差 - Min: {errors.min():.4f}m, Max: {errors.max():.4f}m")
    
    # 计算整体轨迹的差异（逐点）
    all_point_errors = []
    for real_xy, base_xy in zip(all_real, all_base):
        min_len = min(len(real_xy), len(base_xy))
        point_errors = np.sqrt((real_xy[:min_len, 0] - base_xy[:min_len, 0])**2 + 
                               (real_xy[:min_len, 1] - base_xy[:min_len, 1])**2)
        all_point_errors.extend(point_errors.tolist())
    
    all_point_errors = np.array(all_point_errors)
    print(f"\n逐点误差 - Mean: {all_point_errors.mean():.4f}m, Std: {all_point_errors.std():.4f}m")
    print(f"逐点误差 - Min: {all_point_errors.min():.4f}m, Max: {all_point_errors.max():.4f}m")
    
    print("\nAll done!")

if __name__ == "__main__":
    main()
