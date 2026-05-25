import pandas as pd
import numpy as np
import pickle
import os
import glob

def get_unique_filepath(directory, base_name, extension=".pkl"):
    """生成不重复的文件路径，若存在则在名称后加 _n"""
    counter = 1
    out_path = os.path.join(directory, f"{base_name}{extension}")
    while os.path.exists(out_path):
        out_path = os.path.join(directory, f"{base_name}_{counter}{extension}")
        counter += 1
    return out_path

def process_one_csv(csv_path, output_dir):
    """处理单个CSV文件，生成pkl并保存到output_dir（文件名使用第一个时间戳）"""
    # 读取CSV（使用自定义列名，假设所有CSV结构相同）
    custom_names = [
        "timestamp", "throttle", "steering", "leftTicks", "rightTicks",
        "posX", "posY", "posZ", "roll", "pitch", "yaw", "speed",
        "angX", "angY", "angZ", "accX", "accY", "accZ", "cam0", "cam1", "lidar"
    ]
    
    df = pd.read_csv(csv_path, names=custom_names, header=0)
    
    # 时间戳解析（格式：年_月_日_时_分_秒_毫秒）
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format='%Y_%m_%d_%H_%M_%S_%f', errors='coerce')
    df = df.dropna(subset=['timestamp_dt']).reset_index(drop=True)
    
    # 计算相邻时间差（秒）
    df['dt'] = df['timestamp_dt'].diff().dt.total_seconds()
    df['dt'] = df['dt'].replace(0, np.nan)
    
    # 全局速度
    df['vx_global'] = df['posX'].diff() / df['dt']
    df['vy_global'] = df['posY'].diff() / df['dt']
    
    # 转换到车体（yaw为弧度，若实际为度数需转换）
    # df['yaw'] = np.deg2rad(df['yaw'])  # 根据数据实际情况决定是否取消注释
    df['vlon'] = df['vx_global'] * np.cos(df['yaw']) + df['vy_global'] * np.sin(df['yaw'])
    df['vlat'] = -df['vx_global'] * np.sin(df['yaw']) + df['vy_global'] * np.cos(df['yaw'])
    
    # 偏航角速度
    df['vyaw'] = df['yaw'].diff() / df['dt']
    
    # 加速度
    df['acc_lon'] = df['vlon'].diff() / df['dt']
    df['acc_lat'] = df['vlat'].diff() / df['dt']
    df['acc_yaw'] = df['vyaw'].diff() / df['dt']
    df['acc_from_speed'] = df['speed'].diff() / df['dt']
    
    # 删除前两行（差分NaN）
    df_clean = df.iloc[2:].reset_index(drop=True)
    
    # 提取特征
    feature_df = df_clean[['vlon', 'vlat', 'yaw', 'vyaw', 'acc_lon', 'acc_lat', 'acc_yaw']].copy()
    control_df = df_clean[['acc_from_speed','steering','throttle']].copy()
    control_df.columns = ['acc', 'steering','throttle']
    timestamp_series = df_clean['timestamp'].copy()
    
    # 构建数据字典
    data_dict = {
        'timestamp': timestamp_series,
        'features': feature_df,
        'controls': control_df
    }
    
    # 使用第一个时间戳作为文件名
    first_timestamp = str(df_clean['timestamp'].iloc[0])  # 格式如 "2024_01_15_10_30_45_123"
    # 清理非法文件名字符（本格式无非法字符，但保留安全处理）
    safe_name = first_timestamp.replace(':', '_').replace(' ', '_')
    
    # 获取不重复的输出路径
    os.makedirs(output_dir, exist_ok=True)
    out_path = get_unique_filepath(output_dir, safe_name, ".pkl")
    
    with open(out_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"处理完成: {csv_path} -> {out_path} (共 {len(df_clean)} 行)")

if __name__ == "__main__":
    # ========== 配置路径 ==========
    input_dir = r"E:\研究生学习\应用实践论文\DytrProject\Dataset\AutoDRIVE-Nigel-Dataset\slalom_30_hz\ccw"  # 存放所有CSV的根目录（包含多级子目录）
    output_dir = r"E:\研究生学习\应用实践论文\DytrProject\Dataset\read"  # 结果保存目录（所有pkl平铺在此目录下）
    
    # 递归获取所有CSV文件（无论嵌套多少层）
    csv_files = glob.glob(os.path.join(input_dir, "**/*.csv"), recursive=True)
    if not csv_files:
        print(f"未在目录 {input_dir} 中找到任何CSV文件")
        exit(0)
    
    # 批量处理
    for csv_path in csv_files:
        try:
            process_one_csv(csv_path, output_dir)
        except Exception as e:
            print(f"处理文件 {csv_path} 时出错: {e}")
    
    print("所有CSV处理完毕！")