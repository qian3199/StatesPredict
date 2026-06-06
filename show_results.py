import os

# 读取最新输出目录
with open('latest_output_dir.txt', 'r') as f:
    output_dir = f.read().strip()

print(f"Output directory: {output_dir}")

# 列出目录内容
files = os.listdir(output_dir)
print("\nFiles in output directory:")
for file in sorted(files):
    print(f"  - {file}")

# 自动查找最新的验证对比图
val_comparison_files = [f for f in files if 'val_comparison' in f]
val_trajectory_files = [f for f in files if 'val_trajectory' in f]

if val_comparison_files:
    # 按epoch号排序，取最后一个
    val_comparison_files.sort()
    latest_comparison = val_comparison_files[-1]
    print(f"\nLatest validation comparison: {latest_comparison}")
    
if val_trajectory_files:
    val_trajectory_files.sort()
    latest_trajectory = val_trajectory_files[-1]
    print(f"Latest validation trajectory: {latest_trajectory}")

# 生成显示图片的代码
print("\n" + "="*60)
print("Copy the following code to display results:")
print("="*60)

display_code = f"""
from IPython.display import Image
import os

# 读取最新输出目录
with open('latest_output_dir.txt', 'r') as f:
    output_dir = f.read().strip()

# 显示损失曲线
display(Image(filename=os.path.join(output_dir, 'loss_curve.png')))

# 显示验证对比图
display(Image(filename=os.path.join(output_dir, '{latest_comparison}')))

# 显示轨迹对比图
display(Image(filename=os.path.join(output_dir, '{latest_trajectory}')))
"""

print(display_code)
print("="*60)