import pickle
import numpy as np
from scipy.signal import butter, filtfilt
import os
import argparse


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
        self.m = m
        self.Iz = Iz
        self.lf = lf
        self.lr = lr
        self.Caf = Caf
        self.Car = Car
        self.ratio = ratio

    def forward(self, state, control, dt=0.01):
        x, y, vlon, vlat, yaw, omega = state
        acc, steering_angle = control
        delta = steering_angle / self.ratio

        beta = np.arctan2(vlat, vlon) if vlon != 0 else 0

        alpha_f = delta - np.arctan2(self.lf * omega + vlat, vlon) if vlon != 0 else 0
        alpha_r = -np.arctan2(self.lr * omega - vlat, vlon) if vlon != 0 else 0

        Fyf = self.Caf * alpha_f
        Fyr = self.Car * alpha_r

        d_vlon = (Fyf * np.sin(delta) + self.m * vlat * omega + Fyr * np.sin(beta) - Fyf * np.cos(delta)) / self.m + acc
        d_vlat = (-Fyf * np.cos(delta) - Fyr + self.m * vlon * omega) / self.m
        d_omega = (Fyf * self.lf * np.cos(delta) - Fyr * self.lr) / self.Iz

        vlon_new = vlon + d_vlon * dt
        vlat_new = vlat + d_vlat * dt
        omega_new = omega + d_omega * dt

        yaw_new = yaw + omega_new * dt

        v_total = np.sqrt(vlon_new**2 + vlat_new**2)
        if v_total > 0.01:
            x_new = x + v_total * np.cos(yaw_new + np.arctan2(vlat_new, vlon_new)) * dt
            y_new = y + v_total * np.sin(yaw_new + np.arctan2(vlat_new, vlon_new)) * dt
        else:
            x_new = x + vlon_new * np.cos(yaw_new) * dt
            y_new = y + vlon_new * np.sin(yaw_new) * dt

        return np.array([x_new, y_new, vlon_new, vlat_new, yaw_new, omega_new])


class BicycleModelWithNoise(LinearBicycleModel):
    def __init__(self, m=2273.9, Iz=3057.6, lf=1.3535, lr=1.4015, Caf=54920, Car=62190, ratio=15.6, steer_delay_steps=2, process_noise_std=0.0005):
        super().__init__(m, Iz, lf, lr, Caf, Car, ratio)
        self.steer_delay_steps = steer_delay_steps
        self.steer_history = []
        self.process_noise_std = process_noise_std

    def forward(self, state, control, dt=0.01):
        acc, steering_angle = control

        self.steer_history.append(steering_angle)
        if len(self.steer_history) > self.steer_delay_steps:
            self.steer_history.pop(0)
        delayed_steer = np.mean(self.steer_history) if len(self.steer_history) > 0 else steering_angle

        new_control = np.array([acc, delayed_steer])

        # 减小噪声，只对速度和角速度添加小噪声
        noise = np.zeros(6)
        noise[2] = np.random.normal(0, self.process_noise_std)  # vlon
        noise[3] = np.random.normal(0, self.process_noise_std)  # vlat
        noise[5] = np.random.normal(0, self.process_noise_std)  # omega
        noisy_state = state + noise

        result = super().forward(noisy_state, new_control, dt)

        return result


def compute_derivatives(data, dt):
    derivatives = np.zeros_like(data)
    derivatives[0] = (data[1] - data[0]) / dt
    for i in range(1, len(data) - 1):
        derivatives[i] = (data[i+1] - data[i-1]) / (2 * dt)
    derivatives[-1] = (data[-1] - data[-2]) / dt
    return derivatives


def generate_single_left_turn(trip_id, dt=0.01, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    duration = np.random.uniform(5.0, 6.0)
    n_samples = int(duration / dt)

    vlon0 = np.random.uniform(8.0, 15.0)

    controls = np.zeros((n_samples, 2))

    steer_magnitude = np.random.uniform(0.8, 1.2)

    turn_start = int(0.5 / dt)
    turn_end = int(4.5 / dt)
    turn_duration = turn_end - turn_start

    for i in range(n_samples):
        if i < turn_start:
            controls[i, 0] = 0.0
            controls[i, 1] = 0.0
        elif i < turn_end:
            progress = (i - turn_start) / turn_duration
            if progress < 0.15:
                steer = steer_magnitude * (progress / 0.15)
            elif progress < 0.85:
                steer = steer_magnitude
            else:
                steer = steer_magnitude * ((1.0 - progress) / 0.15)
            accel_val = np.random.uniform(-0.05, 0.05) if np.random.random() > 0.9 else 0.0
            controls[i, 0] = accel_val
            controls[i, 1] = steer
        else:
            accel_val = np.random.uniform(-0.05, 0.05) if np.random.random() > 0.9 else 0.0
            controls[i, 0] = accel_val
            controls[i, 1] = 0.0

    m_perturbed = 2273.9 * np.random.uniform(0.98, 1.02)
    Iz_perturbed = 3057.6 * np.random.uniform(0.98, 1.02)
    Caf_perturbed = 54920 * np.random.uniform(0.95, 1.05)
    Car_perturbed = 62190 * np.random.uniform(0.95, 1.05)

    steer_delay = np.random.randint(1, 3)
    process_noise = np.random.uniform(0.0001, 0.001)

    real_model = BicycleModelWithNoise(
        m=m_perturbed,
        Iz=Iz_perturbed,
        lf=1.3535,
        lr=1.4015,
        Caf=Caf_perturbed,
        Car=Car_perturbed,
        ratio=15.6,
        steer_delay_steps=steer_delay,
        process_noise_std=process_noise
    )

    base_model = LinearBicycleModel()

    real_states = np.zeros((n_samples, 6))
    base_states = np.zeros((n_samples, 6))

    x0 = np.random.uniform(-10, 10)
    y0 = np.random.uniform(-10, 10)

    real_states[0] = [x0, y0, vlon0, 0.0, 0.0, 0.0]
    base_states[0] = [x0, y0, vlon0, 0.0, 0.0, 0.0]

    stable = True
    for i in range(n_samples - 1):
        real_states[i+1] = real_model.forward(real_states[i], controls[i], dt=dt)
        base_states[i+1] = base_model.forward(base_states[i], controls[i], dt=dt)

        real_states[i+1, 2] = np.clip(real_states[i+1, 2], 5.0, 50.0)
        real_states[i+1, 3] = np.clip(real_states[i+1, 3], -5.0, 5.0)
        base_states[i+1, 2] = np.clip(base_states[i+1, 2], 5.0, 50.0)
        base_states[i+1, 3] = np.clip(base_states[i+1, 3], -5.0, 5.0)

        if abs(real_states[i+1, 2]) > 40 or abs(real_states[i+1, 3]) > 5:
            stable = False
            break
        if abs(base_states[i+1, 2]) > 40 or abs(base_states[i+1, 3]) > 5:
            stable = False
            break

    if not stable:
        return None

    real_acc_lon = butter_lowpass_filter(compute_derivatives(real_states[:, 2], dt), cutoff_freq=0.01, fs=1/dt, order=1)
    real_acc_lat = butter_lowpass_filter(compute_derivatives(real_states[:, 3], dt), cutoff_freq=0.01, fs=1/dt, order=1)
    real_acc_yaw = butter_lowpass_filter(compute_derivatives(real_states[:, 5], dt), cutoff_freq=0.01, fs=1/dt, order=1)

    real_states_full = np.column_stack([
        real_states[:, 0], real_states[:, 1], real_states[:, 2], real_states[:, 3],
        real_states[:, 4], real_states[:, 5], real_acc_lon, real_acc_lat, real_acc_yaw
    ])

    base_acc_lon = butter_lowpass_filter(compute_derivatives(base_states[:, 2], dt), cutoff_freq=0.01, fs=1/dt, order=1)
    base_acc_lat = butter_lowpass_filter(compute_derivatives(base_states[:, 3], dt), cutoff_freq=0.01, fs=1/dt, order=1)
    base_acc_yaw = butter_lowpass_filter(compute_derivatives(base_states[:, 5], dt), cutoff_freq=0.01, fs=1/dt, order=1)

    base_states_full = np.column_stack([
        base_states[:, 0], base_states[:, 1], base_states[:, 2], base_states[:, 3],
        base_states[:, 4], base_states[:, 5], base_acc_lon, base_acc_lat, base_acc_yaw
    ])

    result = [trip_id, [real_states_full, base_states_full, controls]]
    return result


def generate_left_turn_dataset(output_dir, num_files=500, dt=0.01):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {num_files} left-turn trajectories...")
    print(f"Output directory: {output_dir}")
    print(f"Time step: dt={dt}s")
    print("-" * 50)

    generated = 0
    retry = 0

    while generated < num_files:
        trip_id = f"left_turn_{generated+1:04d}"

        result = generate_single_left_turn(
            trip_id=trip_id,
            dt=dt,
            random_seed=generated * 100 + retry
        )

        if result is not None:
            output_path = os.path.join(output_dir, f"trip_{generated+1:04d}.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(result, f)

            real_states = result[1][0]
            final_yaw = real_states[-1, 4]

            if (generated + 1) % 50 == 0 or generated == 0:
                print(f"[{generated+1}/{num_files}] {trip_id}: {len(real_states)} samples, "
                      f"vlon=[{real_states[:,2].min():.1f}, {real_states[:,2].max():.1f}] m/s, "
                      f"vlat=[{real_states[:,3].min():.2f}, {real_states[:,3].max():.2f}] m/s, "
                      f"final_yaw={final_yaw:.3f} rad")

            generated += 1
            retry = 0
        else:
            retry += 1
            if retry > 50:
                print(f"Warning: Could only generate {generated} valid trajectories")
                break

    print("-" * 50)
    print(f"Done! Generated {generated} files in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate left-turn vehicle dynamics data')
    parser.add_argument('--output_dir', type=str, default='./data/left_turn_dataset',
                        help='Output directory')
    parser.add_argument('--num_files', type=int, default=500,
                        help='Number of trip files to generate')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Time step in seconds')

    args = parser.parse_args()
    generate_left_turn_dataset(args.output_dir, num_files=args.num_files, dt=args.dt)
