import torch
from torch.utils.data import DataLoader
import numpy as np
import glob
import os
import pickle
import logging
import matplotlib
import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(description='DyTR-LSTM 训练脚本')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数 (默认: 50)')
    parser.add_argument('--n_files', type=int, default=500, help='使用的文件数量 (默认: 500)')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小 (默认: 128)')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率 (默认: 5e-5)')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='权重衰减 (默认: 1e-3)')
    parser.add_argument('--hist_len', type=int, default=15, help='历史序列长度 (默认: 15)')
    parser.add_argument('--pred_len', type=int, default=1, help='预测序列长度 (默认: 1)')
    parser.add_argument('--step', type=int, default=1, help='采样步长 (默认: 1)')
    parser.add_argument('--data_dir', type=str, default='./data/left_turn_dataset', help='数据目录 (默认: ./data/left_turn_dataset)')
    parser.add_argument('--output_prefix', type=str, default='DyTR_LSTM', help='输出目录前缀 (默认: DyTR_LSTM)')
    return parser.parse_args()

def run_training():
    args = parse_args()
    
    data_dir = args.data_dir
    pkl_files = sorted(glob.glob(os.path.join(data_dir, '*.pkl')))
    
    logger.info(f"Found {len(pkl_files)} files in {data_dir}")
    
    n_files = min(args.n_files, len(pkl_files))
    n_train = int(n_files * 0.8)
    train_files = pkl_files[:n_train]
    val_files = pkl_files[n_train:n_files]
    
    logger.info(f"Split: {len(train_files)} train, {len(val_files)} val (total {n_files} files)")
    
    hist_len = args.hist_len
    pred_len = args.pred_len
    step = args.step
    
    train_dataset = TimeSeriesDataset(train_files, hist_len=hist_len, pred_len=pred_len, step=step, train=True)
    val_dataset = TimeSeriesDataset(val_files, hist_len=hist_len, pred_len=pred_len, step=step, train=False)
    val_dataset.state_scaler = train_dataset.state_scaler
    val_dataset.control_scaler = train_dataset.control_scaler
    val_dataset.base_scaler = train_dataset.base_scaler
    val_dataset.diff_scaler = train_dataset.diff_scaler
    
    logger.info(f"Train windows: {len(train_dataset.windows)}, Val windows: {len(val_dataset.windows)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = DyTR_LSTM().to(device)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'./outputs/{timestamp}_{args.output_prefix}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    visualizer = VisualUtils(
        state_names=['vlon', 'vlat', 'yaw', 'omega'],
        save_dir=output_dir
    )
    
    trainer = Trainer(
        model=model,
        state_scaler=train_dataset.state_scaler,
        diff_scaler=train_dataset.diff_scaler,
        lr=args.lr,
        weight_decay=args.weight_decay,
        visualizer=visualizer,
        log_dir=output_dir,
        device=device
    )
    
    logger.info(f"Training config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, weight_decay={args.weight_decay}")
    logger.info(f"Data config: n_files={n_files}, hist_len={hist_len}, pred_len={pred_len}, step={step}")
    
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        log_interval=5
    )
    
    logger.info(f"Training completed! Best val loss: {results['best_val_loss']:.6f} at epoch {results['best_epoch']}")
    
    torch.save(model.state_dict(), 'left_turn_model.pth')
    logger.info("Model saved to left_turn_model.pth")

if __name__ == "__main__":
    run_training()