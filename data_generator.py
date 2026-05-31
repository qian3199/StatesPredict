import pickle
import numpy as np
from scipy.signal import butter, filtfilt


def butter_lowpass_filter(data, cutoff_freq, fs=100, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def normalize_angle_sequence(angles):
    normalized = np.zeros_like(angles)
    normalized[0] = normalize_angle(angles[0])
    for i in range(1, len(angles)):
        diff = angles[i] - angles[i-1]
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
        normalized[i] = normalize_angle(normalized[i-1] + diff)
    return normalized


class LinearBicycleModel:
    def __init__(self, m=2273.9, Iz=3057.6, lf=1.3535, lr=1.4015, Caf=54920, Car=62190, ratio=15.6):
        self.m = m      # mass (kg)
        self.Iz = Iz    # yaw moment of inertia (kg·m²)
        self.lf = lf    # distance from CG to front axle (m)
        self.lr = lr    # distance from CG to rear axle (m)
        self.Caf = Caf  # front cornering stiffness (N/rad)
        self.Car = Car  # rear cornering stiffness (N/rad)
        self.ratio = ratio  # steering ratio
    
    def forward(self, state, control, dt=0.01):
        x, y, vlon, vlat, yaw, omega = state
        
        acc, steering_angle = control
        delta = steering_angle / self.ratio
        
        alpha_f = delta - np.arctan2(vlat + self.lf * omega, vlon)
        alpha_r = -np.arctan2(vlat - self.lr * omega, vlon)
        
        Fyf = -self.Caf * alpha_f
        Fyr = -self.Car * alpha_r
        
        dvx_dt = acc + (vlat * omega)
        dvy_dt = (-vlon * omega) + (Fyf + Fyr) / self.m
        domega_dt = (self.lf * Fyf - self.lr * Fyr) / self.Iz
        
        x_new = x + vlon * np.cos(yaw) * dt - vlat * np.sin(yaw) * dt
        y_new = y + vlon * np.sin(yaw) * dt + vlat * np.cos(yaw) * dt
        vlon_new = vlon + dvx_dt * dt
        vlat_new = vlat + dvy_dt * dt
        yaw_new = yaw + omega * dt
        omega_new = omega + domega_dt * dt
        
        return np.array([x_new, y_new, vlon_new, vlat_new, yaw_new, omega_new])


def compute_derivatives(data, dt=0.01):
    acc = np.zeros_like(data)
    acc[1:-1] = (data[2:] - data[:-2]) / (2 * dt)
    acc[0] = acc[1]
    acc[-1] = acc[-2]
    return acc


def process_trip(input_pkl_path, output_pkl_path, dt=0.01, warm_up=20):
    with open(input_pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    trip_id = raw_data['trip_id']
    x = raw_data['x']
    y = raw_data['y']
    vlon = raw_data['vlon']
    vlat = raw_data['vlat']
    yaw = raw_data['yaw']
    omega = raw_data['omega']
    acc = raw_data['acc']
    steering_angle = raw_data['steering_angle']
    
    n_samples = len(x)
    
    filtered_acc = butter_lowpass_filter(acc, cutoff_freq=0.01, fs=1/dt, order=1)
    filtered_steering = butter_lowpass_filter(steering_angle, cutoff_freq=0.01, fs=1/dt, order=1)
    
    controls = np.column_stack([filtered_acc, filtered_steering])
    
    bike_model = LinearBicycleModel()
    
    base_states = np.zeros((n_samples, 6))
    base_states[0] = [x[0], y[0], vlon[0], vlat[0], yaw[0], omega[0]]
    
    for i in range(n_samples - 1):
        base_states[i+1] = bike_model.forward(base_states[i], controls[i], dt=dt)
    
    base_states[:, 4] = normalize_angle_sequence(base_states[:, 4])
    
    real_acc_lon = butter_lowpass_filter(compute_derivatives(vlon, dt), cutoff_freq=0.01, fs=1/dt, order=1)
    real_acc_lat = butter_lowpass_filter(compute_derivatives(vlat, dt), cutoff_freq=0.01, fs=1/dt, order=1)
    real_acc_yaw = butter_lowpass_filter(compute_derivatives(omega, dt), cutoff_freq=0.01, fs=1/dt, order=1)
    
    real_states = np.column_stack([x, y, vlon, vlat, yaw, omega, real_acc_lon, real_acc_lat, real_acc_yaw])
    real_states[:, 4] = normalize_angle_sequence(real_states[:, 4])
    
    base_acc_lon = butter_lowpass_filter(compute_derivatives(base_states[:, 2], dt), cutoff_freq=0.01, fs=1/dt, order=1)
    base_acc_lat = butter_lowpass_filter(compute_derivatives(base_states[:, 3], dt), cutoff_freq=0.01, fs=1/dt, order=1)
    base_acc_yaw = butter_lowpass_filter(compute_derivatives(base_states[:, 5], dt), cutoff_freq=0.01, fs=1/dt, order=1)
    
    base_states_full = np.column_stack([
        base_states[:, 0], base_states[:, 1], base_states[:, 2], base_states[:, 3],
        base_states[:, 4], base_states[:, 5], base_acc_lon, base_acc_lat, base_acc_yaw
    ])
    
    if warm_up > 0:
        real_states = real_states[warm_up:]
        base_states_full = base_states_full[warm_up:]
        controls = controls[warm_up:]
    
    result = [trip_id, [real_states, base_states_full, controls]]
    
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"Processed trip {trip_id}, saved to {output_pkl_path}")
    print(f"Shape: real_states={real_states.shape}, base_states={base_states_full.shape}, controls={controls.shape}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process trip data with linear bicycle model')
    parser.add_argument('--input', required=True, help='Input raw trip pickle file')
    parser.add_argument('--output', required=True, help='Output processed pickle file')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step (default: 0.01)')
    parser.add_argument('--warm_up', type=int, default=20, help='Number of warm-up steps to truncate')
    
    args = parser.parse_args()
    
    process_trip(args.input, args.output, dt=args.dt, warm_up=args.warm_up)
