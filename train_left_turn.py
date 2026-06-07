import torch
from torch.utils.data import DataLoader
import numpy as np
import glob
import os
import pickle
import logging
import matplotlib
import argparse
from datetime import datetime
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import DyTR_LSTM, DyTR_MLP, DyTR_Informer
from dataset import TimeSeriesDataset
from trainer import Trainer
from visualizer import VisualUtils
from inference import InferModel

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='DyTR 训练脚本')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'mlp', 'informer'], 
                        help='选择模型类型: lstm, mlp 或 informer (默认: lstm)')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数 (默认: 50)')
    parser.add_argument('--n_files', type=int, default=500, help='使用的文件数量 (默认: 500)')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小 (默认: 128)')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率 (默认: 5e-5)')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='权重衰减 (默认: 1e-3)')
    parser.add_argument('--hist_len', type=int, default=15, help='历史序列长度 (默认: 15)')
    parser.add_argument('--pred_len', type=int, default=1, help='预测序列长度 (默认: 1)')
    parser.add_argument('--step', type=int, default=1, help='采样步长 (默认: 1)')
    parser.add_argument('--data_dir', type=str, default='./data/left_turn_dataset', help='数据目录 (默认: ./data/left_turn_dataset)')
    parser.add_argument('--output_prefix', type=str, default=None, help='输出目录前缀 (默认: 模型名称)')
    parser.add_argument('--early_stopping', action='store_true', help='启用早停机制 (默认: 禁用)')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值 (默认: 20)')
    parser.add_argument('--min_delta', type=float, default=1e-8, help='早停最小改进阈值 (默认: 1e-8)')
    parser.add_argument('--inference_files', type=int, default=0, help='推理时使用的文件数量，0表示全部 (默认: 0)')
    return parser.parse_args()

def run_training():
    args = parse_args()
    
    # 根据参数选择模型
    model_type = args.model.lower()
    if model_type == 'lstm':
        ModelClass = DyTR_LSTM
        model_name = 'DyTR_LSTM'
    elif model_type == 'mlp':
        ModelClass = DyTR_MLP
        model_name = 'DyTR_MLP'
    elif model_type == 'informer':
        ModelClass = DyTR_Informer
        model_name = 'DyTR_Informer'
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    
    # 设置输出目录前缀
    if args.output_prefix is None:
        output_prefix = model_name
    else:
        output_prefix = args.output_prefix
    
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
    
    # 创建所选模型
    model = ModelClass(hist_len=hist_len, pred_len=pred_len).to(device)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'./outputs/{timestamp}_{output_prefix}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 将输出目录路径保存到文件，方便后续使用
    with open('latest_output_dir.txt', 'w') as f:
        f.write(output_dir)
    logger.info(f"Output directory path saved to: latest_output_dir.txt")
    
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
        device=device,
        early_stopping=args.early_stopping,
        patience=args.patience,
        min_delta=args.min_delta
    )
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Training config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, weight_decay={args.weight_decay}")
    logger.info(f"Data config: n_files={n_files}, hist_len={hist_len}, pred_len={pred_len}, step={step}")
    
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        log_interval=5
    )
    
    logger.info(f"Training completed! Best val loss: {results['best_val_loss']:.6f} at epoch {results['best_epoch']}")
    
    torch.save(model.state_dict(), f'{model_name.lower()}_model.pth')
    logger.info(f"Model saved to {model_name.lower()}_model.pth")
    
    # 保存模型路径配置（方便推理时自动读取）
    save_model_path_config(output_dir, args.model.upper())
    
    # 运行推理 - 使用 Inference_mul.py 的方式
    run_inference_mul(output_dir, args)

def run_inference(output_dir, model, dataset, val_files, inference_files=0):
    """在验证集上运行推理并保存结果"""
    inference_dir = os.path.join(output_dir, 'inference_result')
    os.makedirs(inference_dir, exist_ok=True)
    
    logger.info(f"Running inference, results will be saved to {inference_dir}")
    
    # 创建可视化工具
    inference_visualizer = VisualUtils(
        state_names=['vlon', 'vlat', 'yaw', 'omega'],
        save_dir=inference_dir
    )
    
    # 创建推理模型（不使用物理模型，数据中已包含物理预测）
    infer_model = InferModel(
        model=model,
        state_scaler=dataset.state_scaler,
        diff_scaler=dataset.diff_scaler,
        control_scaler=dataset.control_scaler,
        base_scaler=dataset.base_scaler,
        phy_model=None,
        visualizer=inference_visualizer,
        log_dir=inference_dir
    )
    
    # 在验证集文件上运行推理
    infer_model.batch_inference(
        data_dir='./data/left_turn_dataset',
        output_dir=inference_dir,
        n_files=inference_files
    )
    
    logger.info(f"Inference completed!")

def save_model_path_config(output_dir, model_type):
    """保存最新模型路径到配置文件"""
    import json
    config_path = 'latest_model_config.json'
    
    # 读取现有配置
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # 更新对应模型类型的最新路径
    model_key = f'{model_type}_best_model'
    model_path = os.path.join(output_dir, 'best_model.pth')
    
    config[model_key] = model_path
    config[f'{model_type}_last_model'] = os.path.join(output_dir, 'last_model.pth')
    config[f'{model_type}_output_dir'] = output_dir
    config['latest'] = model_type
    config['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Model path saved to {config_path}: {model_key} = {model_path}")


def run_inference_mul(output_dir, args):
    """使用 Inference_mul.py 的方式运行推理"""
    import subprocess
    
    inference_dir = os.path.join(output_dir, 'inference_mul')
    # 使用与训练保存一致的模型文件名（dytr_xxx_model.pth）
    model_path = f'dytr_{args.model.lower()}_model.pth'
    
    logger.info(f"Running inference_mul, results will be saved to {inference_dir}")
    
    cmd = [
        'python', 'Inference_mul.py',
        '--output_dir', output_dir,
        '--model_path', model_path,
        '--data_dir', args.data_dir,
        '--max_trips', str(args.inference_files if args.inference_files > 0 else 5),
        '--start_idx', str(args.hist_len + 5),
        '--hist_len', str(args.hist_len)
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    logger.info(f"Inference_mul completed! Results saved to {inference_dir}")

if __name__ == "__main__":
    run_training()