
import pickle
import numpy as np
import os


def check_smooth_data():
    # 检查平滑数据
    smooth_file = './data/left_turn_dataset_smoothed/trip_0001.pkl'
    if os.path.exists(smooth_file):
        with open(smooth_file, 'rb') as f:
            smooth_data = pickle.load(f)
        print("平滑数据结构:")
        print(f"  类型: {type(smooth_data)}")
        print(f"  长度: {len(smooth_data)}")
        print(f"  [0]: {type(smooth_data[0])}, {smooth_data[0]}")
        print(f"  [1]: {type(smooth_data[1])}, 长度={len(smooth_data[1])}")
        print(f"  [1][0] shape: {smooth_data[1][0].shape}")
        print(f"  [1][1] shape: {smooth_data[1][1].shape}")
        print(f"  [1][2] shape: {smooth_data[1][2].shape}")


if __name__ == '__main__':
    check_smooth_data()

