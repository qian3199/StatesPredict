
import pickle
import numpy as np
import os


def check_test_data():
    # 检查测试数据
    test_file = './data/left_turn_dataset_smoothed/trip_0000.pkl'
    if os.path.exists(test_file):
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f)
        print("测试数据结构:")
        print(f"  类型: {type(test_data)}")
        print(f"  长度: {len(test_data)}")
        for i in range(len(test_data)):
            print(f"  [{i}]: {type(test_data[i])}, {test_data[i]}")
            if isinstance(test_data[i], np.ndarray):
                print(f"    Shape: {test_data[i].shape}")


if __name__ == '__main__':
    check_test_data()

