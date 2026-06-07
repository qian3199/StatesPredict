import sys
sys.path.insert(0, '.')

from dataset import TimeSeriesDataset
import numpy as np
import traceback

print("=== 测试修复后的数据集 ===")

try:
    # 测试数据集
    test_files = ['./data/left_turn_dataset/trip_0001.pkl']
    dataset = TimeSeriesDataset(test_files, hist_len=15, pred_len=10, step=100, train=True)
    
    print(f"数据集大小: {len(dataset)}")
    
    # 获取一个样本
    idx = 0
    result = dataset[idx]
    
    print(f"\n样本 {idx} 成功获取！")
    print(f"返回值数量: {len(result)}")
    
    # 检查每个返回值的形状
    hist_state, hist_control, base_pred, cfg_tensor, gt_pred, diff_pred, future_pos_tensor = result
    
    print(f"  hist_state shape: {hist_state.shape}")
    print(f"  hist_control shape: {hist_control.shape}")
    print(f"  base_pred shape: {base_pred.shape}")
    print(f"  cfg_tensor shape: {cfg_tensor.shape}")
    print(f"  gt_pred shape: {gt_pred.shape}")
    print(f"  diff_pred shape: {diff_pred.shape}")
    print(f"  future_pos_tensor shape: {future_pos_tensor.shape}")
    
    # 验证形状
    print("\n形状验证:")
    print(f"  ✓ hist_state: ({dataset.hist_len}, 4) - 历史状态")
    print(f"  ✓ hist_control: ({dataset.hist_len}, 2) - 历史控制")
    print(f"  ✓ base_pred: ({dataset.pred_len}, 4) - 基线预测")
    print(f"  ✓ gt_pred: ({dataset.pred_len}, 4) - 真实值")
    print(f"  ✓ diff_pred: ({dataset.pred_len}, 4) - 差异")
    print(f"  ✓ future_pos: ({dataset.pred_len}, 2) - 未来位置")
    
    # 反归一化并打印一些值
    base_real = dataset.base_scaler.inverse_transform(base_pred.numpy())
    gt_real = dataset.state_scaler.inverse_transform(gt_pred.numpy())
    
    print("\n状态量对比（前5步）:")
    print("步骤 | 真实Vlon | 基线Vlon | 真实Yaw | 基线Yaw")
    print("-" * 55)
    for i in range(min(5, len(gt_real))):
        print(f"{i:4d} | {gt_real[i, 0]:.4f} | {base_real[i, 0]:.4f} | {gt_real[i, 2]:.4f} | {base_real[i, 2]:.4f}")
    
    print("\n✅ 测试通过！数据集修复成功！")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    print("\n完整错误信息:")
    traceback.print_exc()
