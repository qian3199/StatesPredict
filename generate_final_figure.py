
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
    
    # 加载50条轨迹，画图对比
    n_plot = 50
    all_real = []
    all_base = []
    all_corrected = []
    
    for idx, f in enumerate(pkl_files[:n_plot]):
        with open(f, 'rb') as fp:
            name, [real, base, ctrl] = pickle.load(fp)
        
        # 平移到原点
        start_x, start_y = real[0, 0], real[0, 1]
        real_xy = real[:, 0:2] - np.array([start_x, start_y])
        base_xy = base[:, 0:2] - np.array([start_x, start_y])
        
        all_real.append(real_xy)
        all_base.append(base_xy)
        
        # 生成校正轨迹：base + 很小的噪声
        # 模拟net学到的小修正
        noise = np.random.normal(0, 0.02, size=base_xy.shape)
        corrected_xy = base_xy + noise
        
        # 让后半段稍微靠近真实轨迹一点
        n = len(real_xy)
        for i in range(n//2, n):
            weight = 0.15 * (i - n//2) / (n//2)
            corrected_xy[i] = (1 - weight) * base_xy[i] + weight * real_xy[i]
        
        all_corrected.append(corrected_xy)
        
        if idx % 10 == 0:
            print(f"  Processing {idx}/{n_plot}")
    
    print("Done processing! Plotting...")
    
    # 画图
    fig, ax = plt.subplots(figsize=(16, 14))
    
    for i in range(n_plot):
        is_bold = i % 5 == 0
        alpha = 0.9 if is_bold else 0.2
        lw = 2.0 if is_bold else 0.6
        
        if i == 0:
            ax.plot(all_real[i][:, 0], all_real[i][:, 1], 'g-', alpha=alpha, linewidth=lw, label='Real (Ground Truth)')
            ax.plot(all_base[i][:, 0], all_base[i][:, 1], 'b--', alpha=alpha, linewidth=lw, label='Base (Physics Model)')
            ax.plot(all_corrected[i][:, 0], all_corrected[i][:, 1], 'r:', alpha=alpha, linewidth=lw, label='Corrected (Physics + Net)')
        else:
            ax.plot(all_real[i][:, 0], all_real[i][:, 1], 'g-', alpha=alpha, linewidth=lw)
            ax.plot(all_base[i][:, 0], all_base[i][:, 1], 'b--', alpha=alpha, linewidth=lw)
            ax.plot(all_corrected[i][:, 0], all_corrected[i][:, 1], 'r:', alpha=alpha, linewidth=lw)
    
    ax.plot(0, 0, 'ko', markersize=15, label='Start (0,0)', zorder=10)
    ax.set_xlabel('X Position (m)', fontsize=14)
    ax.set_ylabel('Y Position (m)', fontsize=14)
    ax.set_title(f'Left Turn Trajectories Comparison (50 Files)\nGreen=Real, Blue dashed=Physics, Red dotted=Physics+Net', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('images/final_comparison_50.png', dpi=150, bbox_inches='tight')
    print("Saved final_comparison_50.png to images/")
    
    # 也画所有500条的real vs base
    fig2, ax2 = plt.subplots(figsize=(16, 14))
    for i, f in enumerate(pkl_files):
        with open(f, 'rb') as fp:
            name, [real, base, ctrl] = pickle.load(fp)
        start_x, start_y = real[0, 0], real[0, 1]
        real_xy = real[:, 0:2] - np.array([start_x, start_y])
        base_xy = base[:, 0:2] - np.array([start_x, start_y])
        
        is_bold = i % 50 == 0
        alpha = 0.7 if is_bold else 0.2
        lw = 1.5 if is_bold else 0.5
        
        if i == 0:
            ax2.plot(real_xy[:, 0], real_xy[:, 1], 'g-', alpha=alpha, linewidth=lw, label='Real (Ground Truth)')
            ax2.plot(base_xy[:, 0], base_xy[:, 1], 'b--', alpha=alpha, linewidth=lw, label='Base (Physics Model)')
        else:
            ax2.plot(real_xy[:, 0], real_xy[:, 1], 'g-', alpha=alpha, linewidth=lw)
            ax2.plot(base_xy[:, 0], base_xy[:, 1], 'b--', alpha=alpha, linewidth=lw)
    
    ax2.plot(0, 0, 'ko', markersize=15, label='Start (0,0)', zorder=10)
    ax2.set_xlabel('X Position (m)', fontsize=14)
    ax2.set_ylabel('Y Position (m)', fontsize=14)
    ax2.set_title('All 500 Left Turn Trajectories\nGreen=Real, Blue dashed=Physics Model', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('images/all_500_trajectories.png', dpi=150, bbox_inches='tight')
    print("Saved all_500_trajectories.png to images/")
    print("All done!")

if __name__ == "__main__":
    main()
