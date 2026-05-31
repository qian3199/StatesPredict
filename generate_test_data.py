import pickle
import numpy as np

def generate_test_data():
    np.random.seed(42)
    
    n_samples = 5000
    dt = 0.01
    t = np.arange(n_samples) * dt
    
    vlon = 20 + 5 * np.sin(0.5 * t) + np.random.randn(n_samples) * 0.5
    vlat = 0.5 * np.sin(1.0 * t) + np.random.randn(n_samples) * 0.3
    yaw = 0.1 + 0.3 * np.sin(0.3 * t) + np.random.randn(n_samples) * 0.05
    omega = 0.05 + 0.2 * np.cos(0.3 * t) + np.random.randn(n_samples) * 0.03
    
    x = np.cumsum(vlon * np.cos(yaw) * dt)
    y = np.cumsum(vlon * np.sin(yaw) * dt)
    
    acc = 1.0 * np.sin(0.2 * t) + np.random.randn(n_samples) * 0.1
    steering_angle = 0.1 * np.sin(0.5 * t) + np.random.randn(n_samples) * 0.02
    
    raw_data = {
        'trip_id': 'test_trip_001',
        'x': x,
        'y': y,
        'vlon': vlon,
        'vlat': vlat,
        'yaw': yaw,
        'omega': omega,
        'acc': acc,
        'steering_angle': steering_angle
    }
    
    with open('raw_trip.pkl', 'wb') as f:
        pickle.dump(raw_data, f)
    
    print(f"Generated raw_trip.pkl with {n_samples} samples")
    
    from data_generator import process_trip
    process_trip('raw_trip.pkl', 'processed_trip.pkl', dt=dt)

if __name__ == "__main__":
    generate_test_data()
