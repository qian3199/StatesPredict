
import numpy as np
import matplotlib.pyplot as plt
import pickle


def test_noise_effect():
    # 创建一个简单的测试来验证噪声对状态量的影响
    dt = 0.01
    n_steps = 500
    time = np.arange(n_steps) * dt
    
    # 模拟无噪声的理想状态
    vlon_clean = np.ones(n_steps) * 10.0
    vlat_clean = np.sin(time * 2) * 0.5  # 转向产生的侧速度
    omega_clean = np.cos(time * 2) * 0.1
    
    # 测试不同噪声水平的影响
    noise_levels = [0.0005, 0.001, 0.005]  # 修复前是0.005，修复后是0.0005
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for i, noise_std in enumerate(noise_levels):
        # 添加噪声
        np.random.seed(42)
        vlon_noisy = vlon_clean + np.random.normal(0, noise_std, n_steps)
        vlat_noisy = vlat_clean + np.random.normal(0, noise_std, n_steps)
        omega_noisy = omega_clean + np.random.normal(0, noise_std, n_steps)
        
        ax = axes[i]
        ax.plot(time, vlon_noisy, 'b-', label=f'Vlon (noise={noise_std})')
        ax.plot(time, vlat_noisy, 'r-', label=f'Vlat (noise={noise_std})')
        ax.plot(time, omega_noisy, 'g-', label=f'Omega (noise={noise_std})')
        ax.set_title(f'噪声标准差 = {noise_std}', fontsize=14)
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 计算波动
        vlon_std = np.std(np.diff(vlon_noisy))
        vlat_std = np.std(np.diff(vlat_noisy))
        omega_std = np.std(np.diff(omega_noisy))
        print(f"噪声={noise_std}: Vlon波动={vlon_std:.6f}, Vlat波动={vlat_std:.6f}, Omega波动={omega_std:.6f}")
    
    plt.tight_layout()
    plt.savefig('noise_effect_comparison.png', dpi=150)
    print("\n噪声效果对比图已保存: noise_effect_comparison.png")
    plt.show()


if __name__ == '__main__':
    test_noise_effect()

