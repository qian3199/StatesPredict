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


if __name__ == "__main__":
    model = DyTR_LSTM()
    
    batch_size = 2
    hist_state = torch.randn(batch_size, 15, 4)
    hist_control = torch.randn(batch_size, 15, 2)
    base_state = torch.randn(batch_size, 10, 4)
    cfg_tensor = torch.FloatTensor([15, 10, 1]).repeat(batch_size, 1)
    gt_pred = torch.randn(batch_size, 10, 4)
    
    model.train()
    delta, corrected = model(hist_state, hist_control, base_state, cfg_tensor, gt_pred)
    
    print(f"delta shape: {delta.shape}")
    print(f"corrected shape: {corrected.shape}")
    
    model.eval()
    delta_eval, corrected_eval = model(hist_state, hist_control, base_state, cfg_tensor)
    
    print(f"delta_eval shape: {delta_eval.shape}")
    print(f"corrected_eval shape: {corrected_eval.shape}")
