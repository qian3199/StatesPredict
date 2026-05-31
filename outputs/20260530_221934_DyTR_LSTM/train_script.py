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
        log_dir=os.path.join(output_dir, 'logs'),
        output_dir=output_dir
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

if __name__ == '__main__':
    run_training()