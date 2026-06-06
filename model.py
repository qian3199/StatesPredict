import torch
import torch.nn as nn
import torch.nn.functional as F


class DyTR_LSTM(nn.Module):
    def __init__(self, state_dim=4, control_dim=2, feature_dim=64, hist_len=15, pred_len=10):
        super(DyTR_LSTM, self).__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.feature_dim = feature_dim
        self.hist_len = hist_len
        self.pred_len = pred_len
        
        self.feature_mlp = nn.Sequential(
            nn.Linear(state_dim + control_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加Dropout
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)   # 添加Dropout
        )
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=2,
            batch_first=True
        )
        
        self.residual_mlp = nn.Sequential(
            nn.Linear(feature_dim + state_dim + 3, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加Dropout
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加Dropout
            nn.Linear(feature_dim, state_dim)
        )
        
        self.teacher_forcing_ratio = 0.5
    
    def forward(self, hist_state, hist_control, base_state, cfg_tensor, gt_pred=None):
        batch_size = hist_state.size(0)
        
        hist_input = torch.cat([hist_state, hist_control], dim=-1)
        features = self.feature_mlp(hist_input)
        
        _, (h_n, _) = self.lstm(features)
        h_n = h_n[-1]
        
        delta_stack = []
        corrected_stack = []
        
        # 动态获取预测长度（从base_state的时间维度）
        pred_len = base_state.size(1)
        
        current_input = base_state[:, 0, :]
        cfg = cfg_tensor[:, :3]
        
        for t in range(pred_len):
            mlp_input = torch.cat([h_n, current_input, cfg], dim=-1)
            delta = self.residual_mlp(mlp_input)
            
            corrected = current_input + delta
            
            delta_stack.append(delta)
            corrected_stack.append(corrected)
            
            if t < pred_len - 1:
                if self.training and gt_pred is not None and torch.rand(1).item() < self.teacher_forcing_ratio:
                    current_input = gt_pred[:, t+1, :]
                else:
                    current_input = corrected.detach()
                
                if base_state.size(1) > 1 and t+1 < base_state.size(1):
                    base_step = base_state[:, t+1, :]
                else:
                    base_step = base_state[:, 0, :]
                
                current_input = base_step + (current_input - base_step)
        
        delta_stack = torch.stack(delta_stack, dim=1)
        corrected_stack = torch.stack(corrected_stack, dim=1)
        
        return delta_stack, corrected_stack


class DyTR_MLP(nn.Module):
    def __init__(self, state_dim=4, control_dim=2, feature_dim=64, hist_len=15, pred_len=10):
        super(DyTR_MLP, self).__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.feature_dim = feature_dim
        self.hist_len = hist_len
        self.pred_len = pred_len
        
        # MLP不具备时序建模能力，需要对历史序列进行特征编码
        # 将历史窗口展平：(batch, hist_len, state_dim+control_dim) -> (batch, hist_len*(state_dim+control_dim))
        self.hist_encoder = nn.Sequential(
            nn.Linear(hist_len * (state_dim + control_dim), feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 残差预测MLP
        self.residual_mlp = nn.Sequential(
            nn.Linear(feature_dim + state_dim + 3, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, state_dim)
        )
        
        self.teacher_forcing_ratio = 0.5
    
    def forward(self, hist_state, hist_control, base_state, cfg_tensor, gt_pred=None):
        batch_size = hist_state.size(0)
        
        # 将历史状态和控制信号拼接并展平
        # hist_state: (batch, hist_len, state_dim)
        # hist_control: (batch, hist_len, control_dim)
        hist_input = torch.cat([hist_state, hist_control], dim=-1)  # (batch, hist_len, state_dim+control_dim)
        
        # 展平历史窗口用于MLP处理
        hist_flat = hist_input.view(batch_size, -1)  # (batch, hist_len*(state_dim+control_dim))
        
        # 编码历史特征
        h_n = self.hist_encoder(hist_flat)  # (batch, feature_dim)
        
        delta_stack = []
        corrected_stack = []
        
        # 动态获取预测长度（从base_state的时间维度）
        pred_len = base_state.size(1)
        
        current_input = base_state[:, 0, :]
        cfg = cfg_tensor[:, :3]
        
        for t in range(pred_len):
            mlp_input = torch.cat([h_n, current_input, cfg], dim=-1)
            delta = self.residual_mlp(mlp_input)
            
            corrected = current_input + delta
            
            delta_stack.append(delta)
            corrected_stack.append(corrected)
            
            if t < pred_len - 1:
                if self.training and gt_pred is not None and torch.rand(1).item() < self.teacher_forcing_ratio:
                    current_input = gt_pred[:, t+1, :]
                else:
                    current_input = corrected.detach()
                
                if base_state.size(1) > 1 and t+1 < base_state.size(1):
                    base_step = base_state[:, t+1, :]
                else:
                    base_step = base_state[:, 0, :]
                
                current_input = base_step + (current_input - base_step)
        
        delta_stack = torch.stack(delta_stack, dim=1)
        corrected_stack = torch.stack(corrected_stack, dim=1)
        
        return delta_stack, corrected_stack


if __name__ == "__main__":
    # Test LSTM
    print("Testing DyTR_LSTM...")
    lstm_model = DyTR_LSTM()
    
    batch_size = 2
    hist_state = torch.randn(batch_size, 15, 4)
    hist_control = torch.randn(batch_size, 15, 2)
    base_state = torch.randn(batch_size, 10, 4)
    cfg_tensor = torch.FloatTensor([15, 10, 1]).repeat(batch_size, 1)
    gt_pred = torch.randn(batch_size, 10, 4)
    
    lstm_model.train()
    delta, corrected = lstm_model(hist_state, hist_control, base_state, cfg_tensor, gt_pred)
    
    print(f"LSTM delta shape: {delta.shape}")
    print(f"LSTM corrected shape: {corrected.shape}")
    
    lstm_model.eval()
    delta_eval, corrected_eval = lstm_model(hist_state, hist_control, base_state, cfg_tensor)
    
    print(f"LSTM delta_eval shape: {delta_eval.shape}")
    print(f"LSTM corrected_eval shape: {corrected_eval.shape}")
    
    # Test MLP
    print("\nTesting DyTR_MLP...")
    mlp_model = DyTR_MLP()
    
    mlp_model.train()
    delta_mlp, corrected_mlp = mlp_model(hist_state, hist_control, base_state, cfg_tensor, gt_pred)
    
    print(f"MLP delta shape: {delta_mlp.shape}")
    print(f"MLP corrected shape: {corrected_mlp.shape}")
    
    mlp_model.eval()
    delta_eval_mlp, corrected_eval_mlp = mlp_model(hist_state, hist_control, base_state, cfg_tensor)
    
    print(f"MLP delta_eval shape: {delta_eval_mlp.shape}")
    print(f"MLP corrected_eval shape: {corrected_eval_mlp.shape}")
