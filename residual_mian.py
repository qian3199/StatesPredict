# residual_main.py
import torch
import torch.nn as nn
import numpy as np
import random
import os
import argparse
import time
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from residual_LSTM import ResidualLSTM
from residual_trainer import ResidualTrainer
from VisualUtils import VisualUtils
from TimeSeriesDataset import TimeSeriesDataset
from log import logger

logger.info("Residual LSTM 训练开始 (残差预测)")

def random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='残差LSTM训练脚本')
    parser.add_argument('--normalize', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    return parser.parse_args()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed()
    args = parse_args()
    
    normalize = args.normalize.lower() == 'true'
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    # ==================== 超参数 ====================
    hist_len, pred_len, step = 15, 10, 16
    state_names = ['v_lon', 'v_lat', 'yaw', 'v_yaw']
    control_names = ['accel', 'steering_angle']
    state_dim, control_dim, config_dim = len(state_names), 2, 1

    filter_flag = True
    feature_dim = 64          # LSTM 隐藏层维度
    layer_nums = 1
    dropout_rate = 0.5

    learning_rate = 1e-4
    weight_decay = 1e-3
    # 残差预测的损失权重 (只关注侧向速度和航向角)
    weights = torch.tensor([0.0, 1.0, 1.0, 0.0], device=device)
    
    val_ratio = 0.2           # 验证集比例
    patience = 30
    delta_loss = 1e-4
    early_step = 1

    # ==================== 数据路径 ====================
    pkl_file = r"E:\研究生学习\应用实践论文\DytrProject\Dataset\predictions\0_prediction.pkl"
    # 结果保存目录
    result_path = f"./results/residual_lstm_{time.strftime('%Y%m%d_%H%M')}"
    checkpoint_path = os.path.join(result_path, "checkpoints")
    img_path = os.path.join(result_path, "img")
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)

    # ==================== 模型 ====================
    model = ResidualLSTM(
        state_dim=state_dim,
        control_dim=control_dim,
        config_dim=config_dim,
        hidden_dim=feature_dim,
        num_layers=layer_nums,
        dropout=dropout_rate,
        T=hist_len,
        pred_len=pred_len
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    visualizer = VisualUtils(state_names, img_path)

    logger.info(f"Device: {device}, Batch size: {batch_size}, Hist: {hist_len}, Pred: {pred_len}, Norm: {normalize}")

    # ==================== 数据集加载（单文件，按样本划分） ====================
    if not os.path.isfile(pkl_file):
        logger.error(f"文件不存在: {pkl_file}")
        exit()

    # 先以 train 模式加载全部数据，会拟合一整套归一化器
    full_dataset = TimeSeriesDataset(
        data_paths=[pkl_file],
        mode='train',
        hist_len=hist_len,
        pred_len=pred_len,
        step=step,
        filter_flag=filter_flag,
        normalize=normalize,
        scalers=None
    )

    if normalize:
        scalers = full_dataset.get_scalers()
        full_dataset.save_scalers(os.path.join(checkpoint_path, "scalers.pkl"))
    else:
        scalers = None

    # 按样本索引划分训练/验证
    total_samples = len(full_dataset)
    indices = list(range(total_samples))
    train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=42)
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset   = Subset(full_dataset, val_idx)

    logger.info(f"总样本数: {total_samples}, 训练: {len(train_dataset)}, 验证: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # ==================== 损失函数 ====================
    class WeightedMSELoss(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.w = w
        def forward(self, pred, target):
            return ((pred - target) ** 2 * self.w).mean()

    criterion = WeightedMSELoss(weights)

    # ==================== Trainer ====================
    # 如果你的 ResidualTrainer 不支持 criterion 参数，请改用 weights=weights
    trainer = ResidualTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        criterion=criterion,        # 若报错，改为 weights=weights，并删除 criterion
        grad_clip_norm=1.0
    )

    # ==================== 训练 ====================
    logger.info("开始训练...")
    train_losses, val_losses = trainer.train(
        num_epochs=num_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=checkpoint_path,
        patience=patience,
        delta_loss=delta_loss,
        early_step=early_step
    )

    # 绘制损失曲线
    visualizer.visual_loss_function(early_step, train_losses, val_losses, label='training_val_loss')
    logger.info("训练完成！")