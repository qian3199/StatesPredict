import torch
import numpy as np
import random
import os
import glob
import argparse
import time

from informer_models_elements.dytr_model import DyTR
from other_model.lstm_dytr import DyTR_LSTM
from other_model.mlp_dytr import DyTR_MLP

from tools.VisualUtils import VisualUtils
from input_data_preprocessing.TimeSeriesDataset import TimeSeriesDataset
# from dytr.dytr_project.residual_model.test_informer_train import Trainer
# from train_Iterative_autoregression import Trainer
# from test_lstm_train import Trainer
from trainer.train_lstm import Trainer
# from train import Trainer



from torch.utils.data import DataLoader
from tools.log import logger

logger.info("DyTR训练开始...")

def random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA卷积操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='DyTR模型训练脚本')
    parser.add_argument('--normalize', type=str, default='true', choices=['true', 'false'],
                       help='是否进行数据归一化 (true/false)')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='批次大小')
    parser.add_argument('--model_flag', type=int, default=3,
                       help='模型选择1: DyTR, 2: DyTR_MLP, 3: DyTR_LSTM')
    return parser.parse_args()

    
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed()
    args = parse_args()
    
    normalize = True if args.normalize.lower() == 'true' else False
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    model_flag = args.model_flag


    hist_len, pred_len, cfg, step = 15, 10, 2273.9, 16
    # dataset_pred_len = 300  
    
    # state_names = ['vlon', 'vlat', 'yaw', 'v_yaw', 'acc_lon', 'acc_lat', 'acc_yaw']
    # state_names = ['acc_lon', 'acc_lat', 'acc_yaw']
    state_names = ['v_lon', 'v_lat', 'yaw', 'v_yaw']
    control_names = ['accel', 'steering_angle']
    state_dim, control_dim, config_dim = len(state_names), 2, 1

    filter_flag = True

    feature_dim = 4
    layer_nums = 1
    pos_kind = "sin"
    dropout_rate = 0.5

    learning_rate = 1e-4
    weight_decay = 1e-3
    # weights =torch.tensor([0, 1.0, 0], device=device)
    # weights = torch.tensor([1.0] * len(state_names), device=device)
    weights =torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
        
    val_ratio = 0.2
    train_ratio = 1 - val_ratio
    dt=0.01
    
    # early stop
    patience = 30
    delta_loss = 1e-4
    early_step = 1 # 验证集评估间隔


    # 数据路径
    result_path =f"/home/luban/dytr/giftdata/dytr_project/residual_model/aqresults/model_{model_flag}_norm_{normalize}_num_epochs{num_epochs}_batch_size{batch_size}_{time.strftime('%Y%m%d_%H%M')}"
    checkpoint_path = f"{result_path}/checkpoints"  
    img_path = f"{result_path}/img"
    voyager_data_path="/home/luban/dytr/giftdata/Data/aqData/all_turn_right/train"
    
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)
    
    if model_flag == 1:
        model = DyTR(
            state_dim=state_dim,     
            control_dim=control_dim, 
            config_dim=config_dim,   
            feature_dim=feature_dim,
            T=hist_len,
            pred_len=pred_len,
            layer_nums=layer_nums,
            pos_kind=pos_kind,
            dropout_rate=dropout_rate
        )
    elif model_flag == 2:
        model = DyTR_MLP(
            state_dim=state_dim,     
            control_dim=control_dim,  
            config_dim=config_dim,   
            hist_len=hist_len,    
            pred_len=pred_len,     
            hidden_dim=feature_dim, 
            num_layers=layer_nums,   
            dropout=dropout_rate
        )
    elif model_flag == 3:
        model = DyTR_LSTM(
            state_dim=state_dim,     
            control_dim=control_dim, 
            config_dim=config_dim,   
            feature_dim=feature_dim,
            T=hist_len,
            pred_len=pred_len,
            layer_nums=layer_nums,
            dropout_rate=dropout_rate,
            teacher_forcing_ratio=1.0
            )
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    visualizer = VisualUtils(state_names, img_path)

    logger.info(f"""=== DyTR参数配置 ===
                \t using device: {device}
                \t batch_size: {batch_size}
                \t 历史长度T: {hist_len}
                \t 预测长度: {pred_len}
                \t cfg(车辆质量): {cfg.shape if hasattr(cfg, 'shape') else cfg}
                \t num epochs: {num_epochs}
                \t learning rate: {learning_rate}
                \t weight decay: {weight_decay}
                \t 验证集比例: {val_ratio}
                \t loss权重: {weights.tolist() if hasattr(weights, 'tolist') else weights}
                \t early stop patience: {patience}
                \t early stop delta loss: {delta_loss}
                \t early stop step: {early_step}
                \t normalize: {normalize}
                \t filter_flag: {filter_flag}
                \t === DyTR模型参数配置 ===
                \t state_names: {state_names}
                \t control_names: {control_names}
                \t state_dim: {state_dim}
                \t control_dim: {control_dim}
                \t config_dim: {config_dim}
                \t feature_dim: {feature_dim}
                \t layer_nums: {layer_nums}
                \t pos_kind: {pos_kind}
                \t dropout_rate: {dropout_rate}
                \t ========================""")
    

    # 获取pickle文件
    pickle_files = glob.glob(os.path.join(voyager_data_path, "*.pkl"))[:1000]
    total_files = len(pickle_files)
    print(f"找到的文件数量: {total_files}")

    if total_files == 0:
        logger.error(f"No pickle files found in {voyager_data_path}")
        exit()
    logger.info(f"Found {total_files} pickle files in {voyager_data_path}")
    
    # 划分训练、验证文件
    train_split = int(total_files * train_ratio)
    train_files = pickle_files[:train_split]
    val_files = pickle_files[train_split:]

    logger.info(f"Using {len(train_files)} files for training, {len(val_files)} files for validation")        

    # 创建训练数据集和数据加载器
    logger.info("创建训练数据集...")
    train_dataset = TimeSeriesDataset(
        data_paths=train_files,
        mode='train',
        hist_len=hist_len,
        pred_len=pred_len,
        step=step,
        filter_flag=filter_flag,
        normalize=normalize,
        scalers=None
    )
    if normalize:
        logger.info("训练数据集归一化...")
        scalers = train_dataset.get_scalers()
        train_dataset.save_scalers(os.path.join(checkpoint_path, "scalers.pkl"))
    else:
        scalers = None

    # 创建验证数据集和数据加载器
    logger.info("创建验证数据集...")
    val_dataset = TimeSeriesDataset(    
        data_paths=val_files,
        mode='val',
        hist_len=hist_len,
        pred_len=pred_len,
        step=step,
        filter_flag=filter_flag,
        normalize=normalize,
        scalers=scalers
        )

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 新增参数
    physics_weights = [1.0, 1.0]   # [vlat直接MSE, 轨迹ADE]
    warm_up_step = 0              # 前50步(0.5秒)不参与积分Loss
    smooth_loss_weight = 0      # 平滑损失权重

    train_model = Trainer(
        model=model,
        opt=opt,
        device=device,
        weights=weights,
        batch_size=batch_size,
        normalize=normalize,
        dt=dt,
        ar_steps=pred_len,
        physics_weights=physics_weights,
        warm_up_step=warm_up_step,
        # smooth_loss_weight=smooth_loss_weight
        initial_tf_ratio=0,   # 初始 100% 开环 (给模型热身)
        tf_decay_epoch=3
    )


    logger.info(f"\n ===========================start trainning=================================")
    
    train_epoch_loss_list, val_epoch_loss_list = train_model.train(
        num_epochs=num_epochs,
        patience=patience,
        delta_loss=delta_loss,
        checkpoint_path=checkpoint_path,
        early_step=early_step,
        visualizer=visualizer,
        train_dataset=train_dataset,
        val_loader=val_loader,
        scalers=scalers
    )
    
    logger.info("All training files processed successfully!")
    visualizer.visual_loss_function(early_step, 
        train_epoch_loss_list, val_epoch_loss_list, label='training_val_loss')
