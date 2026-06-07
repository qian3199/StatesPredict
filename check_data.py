
import pickle
import numpy as np
import os


def check_data_structure():
    # 检查原始数据
    orig_file = './data/left_turn_dataset/trip_0001.pkl'
    if os.path.exists(orig_file):
        with open(orig_file, 'rb') as f:
            orig_data = pickle.load(f)
        print("原始数据结构:")
        print(f"  类型: {type(orig_data)}")
        if isinstance(orig_data, tuple):
            print(f"  长度: {len(orig_data)}")
            for i, item in enumerate(orig_data):
                print(f"  [{i}]: {type(item)}")
                if isinstance(item, np.ndarray):
                    print(f"    Shape: {item.shape}")
                elif isinstance(item, (list, tuple)):
                    print(f"    Length: {len(item)}")
                    for j, subitem in enumerate(item):
                        if isinstance(subitem, np.ndarray):
                            print(f"      [{j}]: {subitem.shape}")
    
    print("\n" + "="*50 + "\n")
    
    # 检查平滑数据
    smooth_file = './data/left_turn_dataset_smoothed/trip_0001.pkl'
    if os.path.exists(smooth_file):
        with open(smooth_file, 'rb') as f:
            smooth_data = pickle.load(f)
        print("平滑数据结构:")
        print(f"  类型: {type(smooth_data)}")
        if isinstance(smooth_data, tuple):
            print(f"  长度: {len(smooth_data)}")
            for i, item in enumerate(smooth_data):
                print(f"  [{i}]: {type(item)}")
                if isinstance(item, np.ndarray):
                    print(f"    Shape: {item.shape}")
                elif isinstance(item, (list, tuple)):
                    print(f"    Length: {len(item)}")
                    for j, subitem in enumerate(item):
                        if isinstance(subitem, np.ndarray):
                            print(f"      [{j}]: {subitem.shape}")


if __name__ == '__main__':
    check_data_structure()

