import torch
from torch.utils.data import DataLoader
import numpy as np
import glob
import os
import pickle
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import DyTR_LSTM
from dataset import TimeSeriesDataset
from trainer import Trainer
from visualizer import VisualUtils
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_training():
    data_dir = './data/left_turn_dataset'
    pkl_files = sorted(glob.glob(os.path.join(data_dir, '*.pkl')))
    
    logger.info(f"Found {len(pkl_files)} files in {data_dir}")
    
    # 只用前50个文件快速训练
    n_files = 50
    n_train = int(n_files * 0.8)
    train_files = pkl_files[:n_train]
    val_files = pkl_files[n_train:n_files]
    
    logger.info(f"Split: {len(train_files)} train, {len(val_files)} val (total {n_files} files)")
    
    hist_len = 15  # 历史序列长度
    pred_len = 1   # 单步预测！
    step = 1       # 每个样本都用
    
    # 使用全部500个文件
    train_dataset = TimeSeriesDataset(train_files, hist_len=hist_len, pred_len=pred_len, step=step, train=True)
    val_dataset = TimeSeriesDataset(val_files, hist_len=hist_len, pred_len=pred_len, step=step, train=False)
    val_dataset.state_scaler = train_dataset.state_scaler
    val_dataset.control_scaler = train_dataset.control_scaler
    val_dataset.base_scaler = train_dataset.base_scaler
    val_dataset.diff_scaler = train_dataset.diff_scaler
    
    logger.info(f"Train windows: {len(train_dataset.windows)}, Val windows: {len(val_dataset.windows)}")
    
    # 使用更大的batch size加快训练
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = DyTR_LSTM().to(device)
    
    # 保存到outputs目录
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'./outputs/{timestamp}_DyTR_LSTM'
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
        lr=5e-5,  # 降低学习率
        weight_decay=1e-3,  # 增加权重衰减（正则化）
        visualizer=visualizer,
        log_dir=output_dir,
        device=device
    )
    
    # 全力训练2个epochs快速测试
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        log_interval=1
    )
    
    logger.info(f"Training completed! Best val loss: {results['best_val_loss']:.4f} at epoch {results['best_epoch']}")
    
    torch.save(model.state_dict(), 'left_turn_model.pth')
    logger.info("Model saved to left_turn_model.pth")

if __name__ == "__main__":
    run_training()