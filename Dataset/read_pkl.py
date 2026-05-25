import pickle
import numpy as np

# 替换成你实际保存的文件路径
pkl_path = r"E:\研究生学习\应用实践论文\DytrProject\Dataset\real_base_control\2023_04_15_02_21_48_368_prediction.pkl"

with open(pkl_path, "rb") as f:
    trip_data = pickle.load(f)

# 解析数据
trip_id, (real_states, pred_states, controls) = trip_data

print(f"Trip ID: {trip_id}")
print(f"真实状态 shape: {real_states.shape}")   # (N, 9) -> [x, y, vlon, vlat, yaw, omega, acc_vlon, acc_vlat, acc_yaw]
print(f"预测状态 shape: {pred_states.shape}")   # (N, 9)
print(f"控制输入 shape: {controls.shape}")      # (N, 2) -> [acc, steering]

# 查看前5行数据
print("\n前5行真实状态:\n", real_states[:5])
print("\n前5行预测状态:\n", pred_states[:5])
print("\n前5行控制输入:\n", controls[:5])

# 如果需要查看某一列（比如纵向速度 vlon，索引2）
vlon_real = real_states[:, 2]
vlon_pred = pred_states[:, 2]
print(f"\n纵向速度真实值范围: {vlon_real.min():.3f} ~ {vlon_real.max():.3f}")
print(f"纵向速度预测值范围: {vlon_pred.min():.3f} ~ {vlon_pred.max():.3f}")

# # 查看数据类型（应该是字典）
# print(type(data))  # <class 'dict'>
# print("字典的键:", data.keys())

# # 分别打印每个键的第一行
# print("\n时间戳第一行:")
# print(data['timestamp'].iloc[0])   # 或 .head(1)

# print("\n状态量第一行:")
# print(data['features'].iloc[0])

# print("\n控制量第一行:")
# print(data['controls'].iloc[0])