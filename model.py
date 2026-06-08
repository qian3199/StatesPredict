import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class ProbAttention(nn.Module):
    """Probabilistic Attention (用于 Informer)"""
    def __init__(self, factor=5, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(self, Q, K, V):
        B, L, H = Q.shape
        _, S, _ = K.shape
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.scale)
        
        # Softmax 和 Dropout
        attn = self.dropout(torch.softmax(scores, dim=-1))
        
        # 应用注意力
        out = torch.matmul(attn, V)
        return out, attn


class InformerLayer(nn.Module):
    """Informer 编码器层"""
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, factor=5):
        super(InformerLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.attention = ProbAttention(factor=factor, scale=d_model, attention_dropout=dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu
    
    def forward(self, x):
        B, L, H = x.shape
        
        # 多头注意力
        q = k = v = x
        attn_out, _ = self.attention(q, k, v)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # FFN
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = x + y
        x = self.norm2(x)
        
        return x


class InformerEncoder(nn.Module):
    """Informer 编码器"""
    def __init__(self, d_model=64, n_heads=8, d_ff=256, n_layers=2, dropout=0.1, factor=5):
        super(InformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            InformerLayer(d_model, n_heads, d_ff, dropout, factor)
            for _ in range(n_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DyTR_Informer(nn.Module):
    def __init__(self, state_dim=4, control_dim=2, feature_dim=64, hist_len=15, pred_len=10):
        super(DyTR_Informer, self).__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.feature_dim = feature_dim
        self.hist_len = hist_len
        self.pred_len = pred_len
        
        # 输入嵌入
        self.input_embedding = nn.Sequential(
            nn.Linear(state_dim + control_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Informer 编码器
        self.encoder = InformerEncoder(
            d_model=feature_dim,
            n_heads=8,
            d_ff=feature_dim * 4,
            n_layers=2,
            dropout=0.1
        )
        
        # 位置编码（可学习）
        self.pos_encoding = nn.Parameter(torch.randn(1, hist_len, feature_dim) * 0.02)
        
        # 预测头
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
        
        # 拼接状态和控制
        hist_input = torch.cat([hist_state, hist_control], dim=-1)  # (B, hist_len, state_dim+control_dim)
        
        # 嵌入
        features = self.input_embedding(hist_input)  # (B, hist_len, feature_dim)
        
        # 添加位置编码
        features = features + self.pos_encoding.expand(batch_size, -1, -1)
        
        # Informer 编码
        encoded = self.encoder(features)  # (B, hist_len, feature_dim)
        
        # 取最后一个时间步的编码作为上下文向量
        h_n = encoded[:, -1, :]  # (B, feature_dim)
        
        delta_stack = []
        corrected_stack = []
        
        # 动态获取预测长度
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
    
    # Test Informer
    print("\nTesting DyTR_Informer...")
    informer_model = DyTR_Informer()
    
    informer_model.train()
    delta_informer, corrected_informer = informer_model(hist_state, hist_control, base_state, cfg_tensor, gt_pred)
    
    print(f"Informer delta shape: {delta_informer.shape}")
    print(f"Informer corrected shape: {corrected_informer.shape}")
    
    informer_model.eval()
    delta_eval_informer, corrected_eval_informer = informer_model(hist_state, hist_control, base_state, cfg_tensor)
    
    print(f"Informer delta_eval shape: {delta_eval_informer.shape}")
    print(f"Informer corrected_eval shape: {corrected_eval_informer.shape}")
    
    print("\nAll models tested successfully!")
