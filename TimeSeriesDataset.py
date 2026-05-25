import numpy as np
from scipy import signal
import pickle
import torch
import joblib
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Optional
from log import logger
from sklearn.preprocessing import StandardScaler 

class TimeSeriesDataset(Dataset):
    """时间序列数据集，滑动窗口"""
    def __init__(self, 
                 data_paths: List[str],
                 mode: str = 'train',
                 hist_len: int = 10,
                 pred_len: int = 1,
                 filter_flag: bool = False,
                 order: int = 1,
                 frequency: float = 0.01, #acc用0.01，其余用0.05
                 step: int = 1,
                 normalize: bool = False,
                 scalers: Optional[Dict[str, StandardScaler]] = None,
                ):
        """
        参数:
            data_paths: 数据文件路径列表
            mode: 模式，'train', 'val'
            hist_len: 历史序列长度
            pred_len: 预测序列长度
            filter_flag: 是否滤波
            order: 滤波阶数
            frequency: 截止频率
            step: 滑动窗口步长
            normalize: 是否归一化
            scalers: 外部传入的归一化器(val)
        """
        self.data_paths = data_paths
        self.mode = mode
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.filter_flag = filter_flag
        self.order = order
        self.frequency = frequency
        self.step = step
        self.normalize = normalize
        
        # 存储样本
        self.samples = []
        
        # 归一化器
        self.state_scaler = None
        self.control_scaler = None
        self.base_scaler = None
        self.fit_scalers = None

        # val提供归一化器
        if mode == 'val':
            self.fit_scalers = False
            if scalers is None and normalize:
                raise ValueError(f"{mode}模式必须提供归一化器（scalers参数）")
            if scalers is not None and normalize:
                self.state_scaler = scalers.get('state')
                self.control_scaler = scalers.get('control')
                self.base_scaler = scalers.get('base')
                self.diff_scaler = scalers.get('diff')
        else:
            self.fit_scalers = True
        
        self._load_and_preprocess()
        
        logger.info(f"{mode}数据集: {len(self.data_paths)}个文件，{len(self.samples)}个样本")
    
    def _load_and_preprocess(self) -> None:
        """加载和预处理所有数据"""
        all_state = []
        all_control = []
        all_base = [] 
        all_diff = []      
        logger.info(f"加载 {len(self.data_paths)} 个文件...")
        
        for file_idx, path in enumerate(self.data_paths):
            try:
                state_data, control_data, base_data = self._load_file(path)            
                diff_data = state_data - base_data
                if file_idx % 200==0:
                    logger.info(f"已加载 {file_idx}/{len(self.data_paths)} 文件")
                if self.normalize and self.fit_scalers:
                    all_state.append(state_data)
                    all_control.append(control_data)
                    all_base.append(base_data)
                    all_diff.append(diff_data)
                
                self._create_samples(state_data, control_data, base_data, diff_data, file_idx)
                
            except Exception as e:
                logger.error(f"加载文件 {path} 失败: {e}")
                continue
        
        if self.normalize:
            if self.fit_scalers:
                all_state = np.vstack(all_state) if all_state else None
                all_control = np.vstack(all_control) if all_control else None
                all_base = np.vstack(all_base) if all_base else None
                all_diff = np.vstack(all_diff) if all_diff else None

                # 初始化StandardScaler
                self.state_scaler = StandardScaler()
                self.control_scaler = StandardScaler()
                self.base_scaler = StandardScaler()
                self.diff_scaler = StandardScaler()
                
                # 拟合
                self.state_scaler.fit(all_state)
                self.control_scaler.fit(all_control)
                self.base_scaler.fit(all_base)
                self.diff_scaler.fit(all_diff)
                
                logger.info(f"批量拟合完成 - 状态: {all_state.shape}, 控制: {all_control.shape}, 基础: {all_base.shape}, 差异: {all_diff.shape}")
                                
            else:
                # val模式：检查归一化器是否存在
                if self.state_scaler is None or self.control_scaler is None or self.base_scaler is None or self.diff_scaler is None:
                    raise ValueError(f"{self.mode}模式未找到归一化器")
                logger.info(f"{self.mode}集使用预训练归一化器")
            
            # 转换所有样本
            self._normalize_samples()
    
    
    def _normalize_samples(self):
        """归一化所有样本"""
        logger.info(f"开始归一化{self.mode}集样本...")
        normalized_count = 0
        
        for sample in self.samples:
            sample['hist_state'] = self.state_scaler.transform(sample['hist_state'])
            sample['hist_control'] = self.control_scaler.transform(sample['hist_control'])
            sample['base_pred'] = self.base_scaler.transform(sample['base_pred'])
            sample['gt_pred'] = self.state_scaler.transform(sample['gt_pred'])
            sample['diff_pred'] = self.diff_scaler.transform(sample['diff_pred'])
            
            # 新增：归一化未来控制量
            sample['u_future'] = self.control_scaler.transform(sample['u_future'])
            
            normalized_count += 1
            
            # 进度日志
            if normalized_count % 50000 == 0:
                logger.info(f"已归一化 {normalized_count}/{len(self.samples)} 个样本")
        logger.info(f"归一化完成")

    def get_scalers(self) -> Dict[str, StandardScaler]:
        """获取归一化器"""
        if self.state_scaler is None or self.control_scaler is None or self.base_scaler is None or self.diff_scaler is None:
            raise ValueError("归一化器未初始化")
        
        return {
            'state': self.state_scaler,
            'control': self.control_scaler,
            'base': self.base_scaler,
            'diff': self.diff_scaler
        }
    
    def save_scalers(self, path: str):
        """保存归一化器"""
        
        scalers = self.get_scalers()
        joblib.dump(scalers, path)
        logger.info(f"归一化器已保存到: {path}")
    
    def _create_samples(self, state_data: np.ndarray, 
                    control_data: np.ndarray,
                    base_data: np.ndarray,
                    diff_data: np.ndarray,
                    file_idx: int) -> None:
        """创建滑动窗口样本"""
        n_samples = len(state_data)
        
        # 确保有足够的数据
        if n_samples < self.hist_len + self.pred_len:
            logger.warning(f"文件 {self.data_paths[file_idx]} 数据不足: {n_samples} < {self.hist_len + self.pred_len}")
            return
        
        # 创建所有起始索引
        start_indices = np.arange(self.hist_len, n_samples - self.pred_len + 1, self.step)
        n_valid = len(start_indices)
        
        # 为每个维度创建索引矩阵
        hist_indices = start_indices.reshape(-1, 1) + np.arange(-self.hist_len, 0)
        pred_indices = start_indices.reshape(-1, 1) + np.arange(self.pred_len)
        
        # 向量化索引获取数据
        hist_state_batch = state_data[hist_indices]  # [n_valid, hist_len, state_dim]
        hist_control_batch = control_data[hist_indices]  # [n_valid, hist_len, control_dim]
        base_pred_batch = base_data[pred_indices]  # [n_valid, pred_len, base_dim]
        diff_pred_batch = diff_data[pred_indices]  # [n_valid, pred_len, diff_dim]
        gt_pred_batch = state_data[pred_indices]  # [n_valid, pred_len, state_dim]
        
        # 新增：提取未来控制量
        u_future_batch = control_data[pred_indices]  # [n_valid, pred_len, control_dim]
        
        # 创建样本列表
        for i in range(n_valid):
            sample = {
                'hist_state': hist_state_batch[i],
                'hist_control': hist_control_batch[i],
                'base_pred': base_pred_batch[i],
                'diff_pred': diff_pred_batch[i],
                'gt_pred': gt_pred_batch[i],
                'u_future': u_future_batch[i],  # 添加到样本字典
                'file_idx': file_idx,
                'time_idx': start_indices[i]
            }
            self.samples.append(sample)
            
    def _load_file(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """加载单个文件"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
                  
        file_id = data[0]
        data_list = data[1]
        ori_state_data = np.array(data_list[0])
        dynamic_data = np.array(data_list[1])
        ori_control_data = np.array(data_list[2])

        # 提取特征
        # x, y,(不作state) vlon, vlat, yaw, v_yaw, acc_lon, acc_lat, acc_yaw
        # accel，steering_angle(1,2)
        # 将控制数据从(1, 2)扩展到(995, 2)
        n_rows = ori_state_data.shape[0]
        if ori_control_data.shape[0] == 1 and n_rows > 1:
            ori_control_data = np.repeat(ori_control_data, n_rows, axis=0)

        state_data = ori_state_data[:, 2:6]
        control_data = ori_control_data[:, [0, 1]]
        base_data = dynamic_data[:, 2:6]
        
        # 应用滤波
        if self.filter_flag:
            state_data = self.filter_data(state_data)
            control_data = self.filter_data(control_data)
            base_data = self.filter_data(base_data)
        
        return state_data, control_data, base_data
    
    def filter_data(self, data: np.ndarray) -> np.ndarray:
        """零相位滤波"""
        if not self.filter_flag or len(data) < 10:
            return data
            
        b, a = signal.butter(self.order, self.frequency)
        n_samples, n_dim = data.shape
        filtered_data = np.zeros_like(data)
        
        for dim in range(n_dim):
            try:
                filtered_data[:, dim] = signal.filtfilt(b, a, data[:, dim])
            except Exception as e:
                logger.warning(f"滤波失败: {e}, 返回原始数据")
                filtered_data[:, dim] = data[:, dim]
                
        return filtered_data
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        sample = self.samples[idx]
        
        # 转换为张量
        hist_state = torch.FloatTensor(sample['hist_state'])
        hist_control = torch.FloatTensor(sample['hist_control'])
        base_pred = torch.FloatTensor(sample['base_pred'])
        gt_pred = torch.FloatTensor(sample['gt_pred'])
        diff_pred = torch.FloatTensor(sample['diff_pred'])
        u_future = torch.FloatTensor(sample['u_future'])  # 新增
        
        # 确保 base_pred 和 gt_pred 具有正确的形状 [pred_len, state_dim]
        if base_pred.ndim == 1:
            base_pred = base_pred.unsqueeze(0)  # 转换为 [1, state_dim]
        if gt_pred.ndim == 1:
            gt_pred = gt_pred.unsqueeze(0)  # 转换为 [1, state_dim]
        
        # 新增：确保 u_future 形状正确
        if u_future.ndim == 1:
            u_future = u_future.unsqueeze(0)  # 转换为 [1, control_dim]

        # 确保形状与模型期望一致
        assert base_pred.shape[0] == self.pred_len, f"base_pred 长度 {base_pred.shape[0]} 不等于 pred_len {self.pred_len}"
        assert gt_pred.shape[0] == self.pred_len, f"gt_pred 长度 {gt_pred.shape[0]} 不等于 pred_len {self.pred_len}"

        # 创建配置张量
        cfg_tensor = torch.zeros(1)  # 占位符
        
        # 返回值末尾增加了 u_future
        return hist_state, hist_control, base_pred, cfg_tensor, gt_pred, diff_pred#, u_future


    def __len__(self):
        return len(self.samples)