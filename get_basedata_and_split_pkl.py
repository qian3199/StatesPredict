import argparse
import pickle
import numpy as np
import os
import glob
from scipy import signal
from dynamic_bicycle import LateralDynamicBicycle

# ========== 车辆参数 ==========
params = {
    "wheelbase": 0.1415,
    "steering_gear_ratio": 1.0,
    "l_f": 0.07075,
    "l_r": 0.07075,
    "c_af": 0.0,
    "c_ar": 0.0,
    "i_z": 0.0,
    "m": 0.0,
}

def normalize_angle_sequence(angles):
    """连续化角度序列（弧度）"""
    angle_norm = angles % (2 * np.pi)
    angle_norm = np.where(angle_norm > np.pi, angle_norm - 2 * np.pi, angle_norm)
    angle_norm = np.where(angle_norm <= -np.pi, angle_norm + 2 * np.pi, angle_norm)
    diff = np.diff(angle_norm)
    diff = np.where(diff > np.pi, diff - 2 * np.pi, diff)
    diff = np.where(diff <= -np.pi, diff + 2 * np.pi, diff)
    angle_norm[1:] = angle_norm[0] + np.cumsum(diff)
    return angle_norm

def process_one_pkl(pkl_path, output_dir, sample_time=None, warm_up=20, vlat_zero_threshold=0.1, filter_freq=0.01):
    """处理单个pkl文件，生成预测结果并保存"""
    # 读取数据
    with open(pkl_path, "rb") as f:
        data_dict = pickle.load(f)
    
    feature_df = data_dict['features']   # vlon, vlat, yaw, vyaw, acc_lon, acc_lat, acc_yaw
    control_df = data_dict['controls']   # acc, steering, throttle
    timestamp_series = data_dict.get('timestamp', None)  # 可选，用于计算采样间隔
    
    # 提取真实状态
    real_vlon = feature_df['vlon'].values
    real_vlat = feature_df['vlat'].values
    real_angle = feature_df['yaw'].values
    real_omega = feature_df['vyaw'].values
    real_acc_vlon = feature_df['acc_lon'].values
    real_acc_vlat = feature_df['acc_lat'].values
    real_acc_angle = feature_df['acc_yaw'].values
    
    # 控制量
    control_acc = control_df['acc'].values
    control_steering = control_df['steering'].values
    
    # 采样时间确定
    if sample_time is None and timestamp_series is not None:
        # 将timestamp转换为datetime（假设格式为字符串，如"2024_01_15_10_30_45_123"）
        try:
            # 尝试解析为 datetime
            if isinstance(timestamp_series.iloc[0], str):
                dt_list = []
                for i in range(len(timestamp_series)-1):
                    t1 = pd.to_datetime(timestamp_series.iloc[i], format='%Y_%m_%d_%H_%M_%S_%f')
                    t2 = pd.to_datetime(timestamp_series.iloc[i+1], format='%Y_%m_%d_%H_%M_%S_%f')
                    dt_list.append((t2 - t1).total_seconds())
                sample_time = np.median(dt_list)
            else:
                sample_time = np.median(np.diff(timestamp_series))
        except:
            sample_time = 0.01
            print(f"警告: {pkl_path} 无法从timestamp计算采样时间，使用默认0.01s")
    elif sample_time is None:
        sample_time = 0.01
        print(f"警告: {pkl_path} 没有timestamp列，使用默认采样时间0.01s")
    
    # 角度连续化
    real_angle = normalize_angle_sequence(real_angle)
    
    # 低通滤波
    b, a = signal.butter(1, filter_freq)
    control_angle = signal.filtfilt(b, a, control_steering)
    control_acc = signal.filtfilt(b, a, control_acc)
    real_vlon_filt = signal.filtfilt(b, a, real_vlon)
    real_vlat_filt = signal.filtfilt(b, a, real_vlat)
    real_omega_filt = signal.filtfilt(b, a, real_omega)
    real_acc_vlon_filt = signal.filtfilt(b, a, real_acc_vlon)
    real_acc_vlat_filt = signal.filtfilt(b, a, real_acc_vlat)
    real_acc_angle_filt = signal.filtfilt(b, a, real_acc_angle)
    
    # 小侧向速度置零
    real_vlat_filt[np.abs(real_vlat_filt) < vlat_zero_threshold] = 0.0
    
    # 从速度重建全局位置（假设起点为原点）
    dt = sample_time
    cos_yaw = np.cos(real_angle)
    sin_yaw = np.sin(real_angle)
    vx_global = real_vlon_filt * cos_yaw - real_vlat_filt * sin_yaw
    vy_global = real_vlon_filt * sin_yaw + real_vlat_filt * cos_yaw
    real_x = np.cumsum(vx_global * dt)
    real_y = np.cumsum(vy_global * dt)
    # 使起点为0
    real_x = np.concatenate(([0.0], real_x[:-1])) if len(real_x) > 0 else np.array([])
    real_y = np.concatenate(([0.0], real_y[:-1])) if len(real_y) > 0 else np.array([])
    
    seq_len = len(real_vlon)
    _t = np.arange(0, seq_len * dt, dt)
    
    # 物理模型预测
    physical_model = LateralDynamicBicycle(parameters=params)
    state_predict = physical_model.State(np.zeros((len(physical_model.States), seq_len)))
    state_real = physical_model.State(np.zeros((len(physical_model.States), seq_len)))
    state_real[0, :] = real_x
    state_real[1, :] = real_y
    state_real[2, :] = real_vlon_filt
    state_real[3, :] = real_vlat_filt
    state_real[4, :] = real_angle
    state_real[5, :] = real_omega_filt
    
    for idx in range(seq_len):
        ctrl = physical_model.ControlInput(np.zeros((len(physical_model.ControlInputs), 1)))
        ctrl[0, 0] = control_acc[idx]
        ctrl[1, 0] = control_angle[idx]
        current_state = state_real[:, idx].reshape(-1, 1)
        new_states, _ = physical_model.simulate(current_state, ctrl, n=2, dt=dt)
        state_predict[:, idx] = new_states[:, -1]
    
    # 提取预测状态
    predict_x = state_predict[0, :]
    predict_y = state_predict[1, :]
    predict_vlon = state_predict[2, :]
    predict_vlat = state_predict[3, :]
    predict_angle = state_predict[4, :]
    predict_omega = state_predict[5, :]
    predict_angle = normalize_angle_sequence(predict_angle)
    
    # 计算预测加速度
    predict_acc_vlon = np.gradient(predict_vlon, _t, edge_order=2)
    predict_acc_vlat = np.gradient(predict_vlat, _t, edge_order=2)
    predict_acc_angle = np.gradient(predict_omega, _t, edge_order=2)
    predict_acc_vlon = signal.filtfilt(b, a, predict_acc_vlon)
    predict_acc_vlat = signal.filtfilt(b, a, predict_acc_vlat)
    predict_acc_angle = signal.filtfilt(b, a, predict_acc_angle)
    predict_vlat[np.abs(predict_vlat) < vlat_zero_threshold] = 0.0
    
    # 丢弃前 warm_up 个点
    warm_up = min(warm_up, seq_len)
    real_x = real_x[warm_up:]
    real_y = real_y[warm_up:]
    real_vlon = real_vlon_filt[warm_up:]
    real_vlat = real_vlat_filt[warm_up:]
    real_angle = real_angle[warm_up:]
    real_omega = real_omega_filt[warm_up:]
    real_acc_vlon = real_acc_vlon_filt[warm_up:]
    real_acc_vlat = real_acc_vlat_filt[warm_up:]
    real_acc_angle = real_acc_angle_filt[warm_up:]
    
    predict_x = predict_x[warm_up:]
    predict_y = predict_y[warm_up:]
    predict_vlon = predict_vlon[warm_up:]
    predict_vlat = predict_vlat[warm_up:]
    predict_angle = predict_angle[warm_up:]
    predict_omega = predict_omega[warm_up:]
    predict_acc_vlon = predict_acc_vlon[warm_up:]
    predict_acc_vlat = predict_acc_vlat[warm_up:]
    predict_acc_angle = predict_acc_angle[warm_up:]
    control_acc = control_acc[warm_up:]
    control_steering = control_steering[warm_up:]
    
    # 组合输出矩阵
    trip_real_states = np.column_stack([
        real_x, real_y, real_vlon, real_vlat, real_angle, real_omega,
        real_acc_vlon, real_acc_vlat, real_acc_angle
    ])
    trip_basemodel_states = np.column_stack([
        predict_x, predict_y, predict_vlon, predict_vlat, predict_angle, predict_omega,
        predict_acc_vlon, predict_acc_vlat, predict_acc_angle
    ])
    control_input = np.column_stack([control_acc, control_steering])
    
    # trip_id: 使用原始文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(pkl_path))[0]
    trip_id = base_name
    
    trip_data = [trip_id, [trip_real_states, trip_basemodel_states, control_input]]
    
    # 保存输出
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{trip_id}_prediction.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(trip_data, f)
    
    print(f"已处理: {pkl_path} -> {out_path} (长度: {len(real_x)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量处理pkl文件，使用动力学模型预测并保存结果")
    parser.add_argument("--input_dir", type=str, required=True, help="包含输入pkl文件的目录（会递归查找）")
    parser.add_argument("--output_dir", type=str, default="./real_base_control", help="保存预测pkl的目录（默认./real_base_control）")
    parser.add_argument("--sample_time", type=float, default=None, help="固定采样时间（秒），若不指定则从timestamp列自动计算")
    parser.add_argument("--warm_up", type=int, default=20, help="丢弃前N个点（默认20）")
    parser.add_argument("--filter_freq", type=float, default=0.01, help="低通滤波截止频率（Hz）")
    args = parser.parse_args()
    
    # 递归查找所有.pkl文件
    pkl_files = glob.glob(os.path.join(args.input_dir, "**/*.pkl"), recursive=True)
    if not pkl_files:
        print(f"在目录 {args.input_dir} 中未找到任何.pkl文件")
        exit(0)
    
    for pkl_path in pkl_files:
        try:
            process_one_pkl(pkl_path, args.output_dir,
                            sample_time=args.sample_time,
                            warm_up=args.warm_up,
                            filter_freq=args.filter_freq)
        except Exception as e:
            print(f"处理文件 {pkl_path} 时出错: {e}")
    
    print("所有pkl处理完毕！")


##python E:\研究生学习\应用实践论文\DytrProject\get_basedata_and_split_pkl.py --input_dir E:\研究生学习\应用实践论文\DytrProject\Dataset\read --output_dir E:\研究生学习\应用实践论文\DytrProject\Dataset\real_base_control