import torch
import torch.nn as nn

class DyTR_LSTM(nn.Module):
    def __init__(self, state_dim, control_dim, config_dim=1, feature_dim=64,
                 T=15, pred_len=1, layer_nums=2, dropout_rate=0.2,
                 teacher_forcing_ratio=1.0, **kwargs):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.config_dim = config_dim
        self.feature_dim = feature_dim
        self.T = T
        self.pred_len = pred_len
        self.dropout_rate = dropout_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio
        # self.output_activation = nn.Tanh()  # 确保输出在[-1, 1]范围内
        
        # self.delta_scale = kwargs.get('delta_scale', 5.0)

        # 特征提取 MLP（每个时间步的 [state, control] -> feature_dim）
        input_dim = state_dim + control_dim
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # LSTM 时序融合
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=layer_nums,
            batch_first=True,
            dropout=dropout_rate if layer_nums > 1 else 0
        )

        # 残差估计 MLP（拼接 LSTM 最后隐藏状态 + 未来状态 + 配置）
        query_dim = state_dim + config_dim
        self.residual_mlp = nn.Sequential(
            nn.Linear(feature_dim + query_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, state_dim)
        )

        self._init_weights()

    def _init_weights(self):
        # LSTM 初始化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        # MLP 初始化
        for module in [self.feature_mlp, self.residual_mlp]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                        
    def forward(self, s_hist, u_hist, s_next=None, c=None, base_state=None, return_corrected=True, teacher_forcing_ratio=None):
        batch_size = s_hist.size(0)

        # 1. 特征提取
        x = torch.cat([s_hist, u_hist], dim=-1)
        x = self.feature_mlp(x)

        # 2. LSTM 编码
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]

        # 3. 动态确定预测步数
        if s_next is not None:
            if s_next.dim() == 2:
                s_next = s_next.unsqueeze(1)
            actual_pred_len = s_next.size(1)
        else:
            actual_pred_len = self.pred_len

        # 4. 处理 c
        if c is None:
            c = torch.zeros(batch_size, self.config_dim, device=h_last.device)
        elif c.dim() == 1:
            c = c.unsqueeze(-1)

        if teacher_forcing_ratio is None:
            tf_ratio = self.teacher_forcing_ratio if self.training else 0.0
        else:
            tf_ratio = teacher_forcing_ratio

        # 5. 准备初始输入
        if base_state is not None:
            if base_state.dim() == 2:
                base_state = base_state.unsqueeze(1)
            current_input = base_state[:, 0, :]
        else:
            current_input = s_hist[:, -1, :]

        outputs = []
        deltas = []

        for t in range(actual_pred_len):
            query = torch.cat([current_input, c], dim=-1)
            fused = torch.cat([h_last, query], dim=-1)

            delta = self.residual_mlp(fused) 
            # delta = self.delta_scale * self.output_activation(delta)

            corrected = current_input + delta

            outputs.append(corrected)
            deltas.append(delta)

            # Teacher Forcing / Autoregressive 逻辑
            if t < actual_pred_len - 1:
                if self.training and (torch.rand(1).item() < tf_ratio):
                    current_input = s_next[:, t + 1, :]
                else:
                    # 自回归模式：用修正后的结果作为下一步输入
                    current_input = corrected.detach()

        deltas_stack = torch.stack(deltas, dim=1)
        outputs_stack = torch.stack(outputs, dim=1)

        if return_corrected:
            return deltas_stack, outputs_stack
        else:
            return deltas_stack, deltas_stack