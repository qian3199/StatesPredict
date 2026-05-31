import torch
from torch.utils.data import DataLoader
import numpy as np
import glob
import os
import pickle
import logging
import shutil
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import DyTR_LSTM
from dataset import TimeSeriesDataset
from trainer import Trainer
from visualizer import VisualUtils

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def compute_xy_trajectory_from_states(states, initial_pos=None, initial_yaw=None):
    if states.ndim == 3:
        B, T, D = states.shape
        vlon = states[:, :, 0]
        vlat = states[:, :, 1]
        yaw = states[:, :, 2]
    else:
        T, D = states.shape
        B = 1
        vlon = states[:, 0].unsqueeze(0) if isinstance(states, torch.Tensor) else states[:, 0:1]
        vlat = states[:, 1].unsqueeze(0) if isinstance(states, torch.Tensor) else states[:, 1:2]
        yaw = states[:, 2].unsqueeze(0) if isinstance(states, torch.Tensor) else states[:, 2:3]

    vx = vlon * np.cos(yaw) - vlat * np.sin(yaw)
    vy = vlon * np.sin(yaw) + vlat * np.cos(yaw)

    if isinstance(states, torch.Tensor):
        dx = torch.cumsum(vx, dim=1) * 0.01
        dy = torch.cumsum(vy, dim=1) * 0.01
        dx = torch.cat([torch.zeros(B, 1, device=dx.device), dx], dim=1)
        dy = torch.cat([torch.zeros(B, 1, device=dy.device), dy], dim=1)
    else:
        dx = np.cumsum(vx, axis=1)
        dy = np.cumsum(vy, axis=1)
        dx = np.concatenate([np.zeros((B, 1)), dx], axis=1)
        dy = np.concatenate([np.zeros((B, 1)), dy], axis=1)

    if initial_pos is not None:
        dx = dx + initial_pos[0]
        dy = dy + initial_pos[1]

    xy = np.stack([dx, dy], axis=-1)
    return xy


def plot_500_trajectories_comparison(data_dir, model, state_scaler, output_path):
    pkl_files = sorted(glob.glob(os.path.join(data_dir, '*.pkl')))
    logger.info(f"Generating trajectories for {len(pkl_files)} files...")

    all_real_x = []
    all_real_y = []
    all_base_x = []
    all_base_y = []
    all_net_x = []
    all_net_y = []

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for idx, pkl_path in enumerate(pkl_files):
            if idx % 100 == 0:
                logger.info(f"Processing {idx}/{len(pkl_files)}...")

            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            states = data['states']
            controls = data['controls']
            initial_x = data['initial_x']
            initial_y = data['initial_y']
            initial_yaw = data['initial_yaw']

            n_windows = max(1, (len(states) - 15) // 4)

            for w_idx in range(min(n_windows, 10)):
                start_idx = w_idx * 4
                end_idx = start_idx + 15

                hist_len = 15
                pred_len = 10

                if end_idx + pred_len > len(states):
                    continue

                hist_state = states[start_idx:end_idx]
                hist_control = controls[start_idx:end_idx]
                base_state = states[end_idx:end_idx + pred_len]
                gt_state = states[end_idx:end_idx + pred_len]

                hist_state_norm = state_scaler.transform(hist_state)
                base_state_norm = state_scaler.transform(base_state)

                hist_state_t = torch.FloatTensor(hist_state_norm).unsqueeze(0).to(device)
                base_state_t = torch.FloatTensor(base_state_norm).unsqueeze(0).to(device)
                hist_control_t = torch.FloatTensor(hist_control).unsqueeze(0).to(device)
                cfg_t = torch.FloatTensor([[hist_len, pred_len, 1.0]]).to(device)

                delta_norm, corrected_norm = model(hist_state_t, hist_control_t, base_state_t, cfg_t)

                corrected_np = corrected_norm.cpu().numpy()[0]
                corrected_denorm = state_scaler.inverse_transform(corrected_np)
                base_denorm = base_state

                if idx == 0 and w_idx == 0:
                    real_xy = compute_xy_trajectory_from_states(
                        torch.FloatTensor(gt_state).unsqueeze(0),
                        initial_pos=(initial_x, initial_y)
                    )[0]
                    base_xy = compute_xy_trajectory_from_states(
                        torch.FloatTensor(base_denorm).unsqueeze(0),
                        initial_pos=(initial_x, initial_y)
                    )[0]
                    net_xy = compute_xy_trajectory_from_states(
                        torch.FloatTensor(corrected_denorm).unsqueeze(0),
                        initial_pos=(initial_x, initial_y)
                    )[0]

                    all_real_x.extend(real_xy[:, 0])
                    all_real_y.extend(real_xy[:, 1])
                    all_base_x.extend(base_xy[:, 0])
                    all_base_y.extend(base_xy[:, 1])
                    all_net_x.extend(net_xy[:, 0])
                    all_net_y.extend(net_xy[:, 1])

    plt.figure(figsize=(14, 10))

    plt.plot(all_real_x, all_real_y, 'g-', label='Real Trajectory', alpha=0.6, linewidth=1)
    plt.plot(all_base_x, all_base_y, 'b--', label='Physics Base', alpha=0.6, linewidth=1)
    plt.plot(all_net_x, all_net_y, 'r-', label='Net+Base (Corrected)', alpha=0.6, linewidth=1)

    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Y (m)', fontsize=12)
    plt.title('500 Trajectories: Real vs Physics Base vs Net+Base', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved 500 trajectories comparison to {output_path}")


def run_training():
    data_dir = './data/left_turn_dataset_v2'
    pkl_files = sorted(glob.glob(os.path.join(data_dir, '*.pkl')))

    logger.info(f"Found {len(pkl_files)} files in {data_dir}")

    n_train = int(len(pkl_files) * 0.8)
    train_files = pkl_files[:n_train]
    val_files = pkl_files[n_train:]

    logger.info(f"Split: {len(train_files)} train, {len(val_files)} val")

    hist_len = 15
    pred_len = 10
    step = 4

    train_dataset = TimeSeriesDataset(train_files, hist_len=hist_len, pred_len=pred_len, step=step, train=True)
    val_dataset = TimeSeriesDataset(val_files, hist_len=hist_len, pred_len=pred_len, step=step, train=False)
    val_dataset.state_scaler = train_dataset.state_scaler
    val_dataset.control_scaler = train_dataset.control_scaler
    val_dataset.base_scaler = train_dataset.base_scaler
    val_dataset.diff_scaler = train_dataset.diff_scaler

    logger.info(f"Train windows: {len(train_dataset.windows)}, Val windows: {len(val_dataset.windows)}")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = DyTR_LSTM().to(device)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = 'DyTR_LSTM'
    run_name = f'{timestamp}_{model_name}'
    output_dir = f'./outputs/{run_name}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)

    logger.info(f'Output directory: {output_dir}')

    visualizer = VisualUtils(
        state_names=['vlon', 'vlat', 'yaw', 'omega'],
        save_dir=os.path.join(output_dir, 'plots')
    )

    trainer = Trainer(
        model=model,
        state_scaler=train_dataset.state_scaler,
        diff_scaler=train_dataset.diff_scaler,
        visualizer=visualizer,
        log_dir=os.path.join(output_dir, 'logs')
    )

    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        log_interval=5
    )

    logger.info(f"Training completed! Best val loss: {results['best_val_loss']:.4f} at epoch {results['best_epoch']}")

    model_path = os.path.join(output_dir, 'left_turn_model.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    src_train_script = os.path.join(output_dir, 'train_script.py')
    shutil.copy(__file__, src_train_script)
    logger.info(f"Training script copied to {src_train_script}")

    logger.info("Generating 500 trajectories comparison plot...")
    plot_500_trajectories_comparison(
        data_dir=data_dir,
        model=model,
        state_scaler=train_dataset.state_scaler,
        output_path=os.path.join(output_dir, 'all_500_trajectories.png')
    )

    return output_dir


if __name__ == '__main__':
    output_dir = run_training()
    print(f"\nTraining finished! Output directory: {output_dir}")