import torch
import torch.nn as nn

class ResidualLSTM(nn.Module):
    """
    输入:
        s_hist:   [batch, T, state_dim]    历史真实状态
        u_hist:   [batch, T, control_dim]  历史控制
        base:     [batch, pred_len, state_dim]  物理模型预测的未来状态
        c:        [batch, config_dim]      可选物理参数（如质量）
    输出:
        residuals:   [batch, pred_len, state_dim]  预测残差
        predictions: [batch, pred_len, state_dim]  预测状态 = base + residuals
    """
    def __init__(self, state_dim, control_dim, config_dim=1, hidden_dim=128,
                 num_layers=2, dropout=0.2, T=15, pred_len=10):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.config_dim = config_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.T = T
        self.pred_len = pred_len

        # 编码器：将历史状态+控制映射到隐状态
        encoder_input_dim = state_dim + control_dim
        self.encoder_fc = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 解码器：每个时间步预测残差
        # 输入：LSTM最后隐状态 + 当前步的物理模型预测状态 + 配置参数
        decoder_input_dim = hidden_dim + state_dim + config_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, s_hist, u_hist, base, c=None):
        batch_size = s_hist.size(0)
        
        # ---------- 维度检查与修正 ----------
        # 确保 base 是 [B, pred_len, state_dim]
        if base.dim() == 2:
            # 可能是 [B, state_dim] 或 [B, pred_len]？假设是 [B, state_dim]
            base = base.unsqueeze(1).expand(-1, self.pred_len, -1)   # [B,1,state] -> [B,pred_len,state]
        elif base.dim() == 3:
            # 已经是预期形状，但检查第二维长度
            if base.size(1) != self.pred_len:
                # 如果第二维是1，可能需要扩展
                if base.size(1) == 1:
                    base = base.expand(-1, self.pred_len, -1)
                else:
                    raise ValueError(f"base 第二维长度 {base.size(1)} 与 pred_len {self.pred_len} 不匹配")
        else:
            raise ValueError(f"base 维度应为2或3，实际为 {base.dim()}")
        
        # 确保 c 是 [B, pred_len, config_dim]
        if c is None:
            c = torch.zeros(batch_size, self.config_dim, device=s_hist.device)
            c = c.unsqueeze(1).expand(-1, self.pred_len, -1)  # [B,1,config] -> [B,pred_len,config]
        else:
            if c.dim() == 1:
                # [config_dim] 或 [batch]？假设是标量扩展
                if c.size(0) == self.config_dim:
                    c = c.unsqueeze(0).unsqueeze(1).expand(batch_size, self.pred_len, -1)
                else:
                    # 可能是 [batch]
                    c = c.unsqueeze(1).expand(-1, self.pred_len).unsqueeze(-1)
            elif c.dim() == 2:
                # [B, config_dim] -> [B, 1, config_dim] -> [B, pred_len, config_dim]
                c = c.unsqueeze(1).expand(-1, self.pred_len, -1)
            elif c.dim() == 3:
                if c.size(1) == 1:
                    c = c.expand(-1, self.pred_len, -1)
                elif c.size(1) != self.pred_len:
                    raise ValueError(f"c 第二维长度 {c.size(1)} 与 pred_len {self.pred_len} 不匹配")
            else:
                raise ValueError(f"c 维度应为1,2或3，实际为 {c.dim()}")
        
        # ---------- 编码 ----------
        x = torch.cat([s_hist, u_hist], dim=-1)          # [B, T, state+ctrl]
        x = self.encoder_fc(x)                           # [B, T, hidden]
        _, (h_n, _) = self.lstm(x)                       # h_n: [num_layers, B, hidden]
        last_hidden = h_n[-1]                            # [B, hidden]

        # 扩展隐状态到每个预测步
        last_hidden_expanded = last_hidden.unsqueeze(1).expand(-1, self.pred_len, -1)  # [B, pred_len, hidden]

        print(f"last_hidden_expanded shape: {last_hidden_expanded.shape}")
        print(f"base shape: {base.shape}")
        print(f"c shape: {c.shape}")
        print(f"decoder_input_dim expected: {self.decoder[0].in_features}")

        # 解码器输入
        decoder_input = torch.cat([last_hidden_expanded, base, c], dim=-1)  # [B, pred_len, hidden+state+config]
        residuals = self.decoder(decoder_input)                # [B, pred_len, state_dim]
        predictions = base + residuals                         # 修正后的预测状态

        return residuals, predictions