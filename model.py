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


class ProbAttention(nn.Module):
    """ProbSparse Self-Attention - Informer的核心组件"""
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        
    def _prob_Q(self, Q, K):
        """计算ProbSparse注意力"""
        B, H, L, E = Q.shape  # Batch, Head, Length, Embedding
        _, _, S, _ = K.shape  # S = seq_len for K
        
        # 计算 Q 的稀疏性度量
        Q_expand = Q.unsqueeze(-2)  # (B, H, L, 1, E)
        K_expand = K.unsqueeze(-3)  # (B, H, 1, S, E)
        
        # 计算注意力分数
        attn_weight = torch.sum(Q_expand * K_expand, dim=-1) / (E ** 0.5)  # (B, H, L, S)
        
        # 选择 top-k 个关键位置
        if self.factor > 0 and L > self.factor:
            # 简化版本：使用 torch.topk 选择重要位置
            top_k = min(self.factor, S)
            _, top_idx = torch.topk(attn_weight, k=top_k, dim=-1)  # (B, H, L, top_k)
            
            # 构建稀疏注意力
            sparse_attn = torch.zeros_like(attn_weight)
            sparse_attn.scatter_(-1, top_idx, torch.gather(attn_weight, -1, top_idx))
        else:
            sparse_attn = attn_weight
            
        return sparse_attn
        
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        # 计算稀疏注意力
        scores = self._prob_Q(queries.transpose(1, 2), keys.transpose(1, 2))  # (B, H, L, S)
        
        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask.unsqueeze(1), -1e9)
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力到 values
        context = torch.matmul(attn, values.transpose(1, 2))  # (B, H, L, D)
        context = context.transpose(1, 2)  # (B, L, H, D)
        
        return context.contiguous(), attn


class FullAttention(nn.Module):
    """标准全注意力"""
    def __init__(self, mask_flag=True, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        # 缩放点积注意力
        scale = E ** -0.5
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale
        
        if self.mask_flag:
            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1), -1e9)
                
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum("bhls,bshd->blhd", attn, values)
        return out.contiguous(), attn


class AttentionLayer(nn.Module):
    """多头注意力层"""
    def __init__(self, d_model, n_heads, attention='prob', factor=5, dropout=0.1):
        super(AttentionLayer, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        if attention == 'prob':
            self.attention = ProbAttention(mask_flag=True, factor=factor, attention_dropout=dropout)
        else:
            self.attention = FullAttention(mask_flag=True, attention_dropout=dropout)
            
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, attn_mask=None):
        B, L, _ = Q.shape
        _, S, _ = K.shape
        
        # 线性变换 + 分头
        Q = self.W_Q(Q).view(B, L, self.n_heads, self.d_k)
        K = self.W_K(K).view(B, S, self.n_heads, self.d_k)
        V = self.W_V(V).view(B, S, self.n_heads, self.d_k)
        
        # 注意力计算
        out, attn = self.attention(Q, K, V, attn_mask)
        
        # 合并多头
        out = out.view(B, L, self.n_heads * self.d_k)
        out = self.W_O(out)
        
        return out, attn


class EncoderLayer(nn.Module):
    """Informer 编码器层"""
    def __init__(self, d_model, n_heads, attention='prob', factor=5, dropout=0.1, d_ff=None):
        super(EncoderLayer, self).__init__()
        
        d_ff = d_ff or 4 * d_model
        self.attention = AttentionLayer(d_model, n_heads, attention, factor, dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu
        
    def forward(self, x, attn_mask=None):
        # Multi-head attention
        attn_out, _ = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feed-forward with conv
        residual = x
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.conv2(self.dropout(self.activation(self.conv1(x))))
        x = x.transpose(1, 2)  # (B, L, D)
        x = x + self.dropout(residual)
        x = self.norm2(x)
        
        return x


class DecoderLayer(nn.Module):
    """Informer 解码器层"""
    def __init__(self, d_model, n_heads, attention='prob', factor=5, dropout=0.1, d_ff=None):
        super(DecoderLayer, self).__init__()
        
        d_ff = d_ff or 4 * d_model
        self.self_attention = AttentionLayer(d_model, n_heads, attention, factor, dropout)
        self.cross_attention = AttentionLayer(d_model, n_heads, attention, factor, dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu
        
    def forward(self, x, encoder_out, self_attn_mask=None, cross_attn_mask=None):
        # Self attention
        self_attn_out, _ = self.self_attention(x, x, x, self_attn_mask)
        x = x + self.dropout(self_attn_out)
        x = self.norm1(x)
        
        # Cross attention
        cross_attn_out, _ = self.cross_attention(x, encoder_out, encoder_out, cross_attn_mask)
        x = x + self.dropout(cross_attn_out)
        x = self.norm2(x)
        
        # Feed-forward
        residual = x
        x = x.transpose(1, 2)
        x = self.conv2(self.dropout(self.activation(self.conv1(x))))
        x = x.transpose(1, 2)
        x = x + self.dropout(residual)
        x = self.norm3(x)
        
        return x


class ConvLayer(nn.Module):
    """Informer 的 Distilling 层"""
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=1,
            padding_mode='circular'
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.downConv(x.transpose(1, 2))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class InformerEncoder(nn.Module):
    """Informer 编码器"""
    def __init__(self, d_model, n_heads, n_layers, attention='prob', factor=5, dropout=0.1, d_ff=None):
        super(InformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, attention, factor, dropout, d_ff)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


class InformerDecoder(nn.Module):
    """Informer 解码器"""
    def __init__(self, d_model, n_heads, n_layers, attention='prob', factor=5, dropout=0.1, d_ff=None):
        super(InformerDecoder, self).__init__()
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, attention, factor, dropout, d_ff)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, encoder_out, self_attn_mask=None, cross_attn_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_out, self_attn_mask, cross_attn_mask)
        return x


class DyTR_Informer(nn.Module):
    """
    基于 Informer 的 DyTR 模型
    使用 Transformer/Informer 架构进行时序建模
    """
    def __init__(self, state_dim=4, control_dim=2, feature_dim=64, hist_len=15, pred_len=10,
                 d_model=64, n_heads=4, n_encoder_layers=2, n_decoder_layers=1, 
                 attention='prob', factor=5, dropout=0.1, d_ff=None):
        super(DyTR_Informer, self).__init__()
        
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.feature_dim = feature_dim
        self.hist_len = hist_len
        self.pred_len = pred_len
        
        # 输入嵌入（编码器用）
        self.input_embedding = nn.Sequential(
            nn.Linear(state_dim + control_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 解码器输入嵌入（只接受 state_dim）
        self.decoder_embedding = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.randn(1, hist_len + pred_len, d_model) * 0.02)
        
        # Informer 编码器和解码器
        self.encoder = InformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            attention=attention,
            factor=factor,
            dropout=dropout,
            d_ff=d_ff
        )
        
        self.decoder = InformerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            attention=attention,
            factor=factor,
            dropout=dropout,
            d_ff=d_ff
        )
        
        # 输出投影：从 d_model 映射到 state_dim
        self.output_proj = nn.Linear(d_model, state_dim)
        
        # 残差 MLP（用于 refinement）
        self.residual_mlp = nn.Sequential(
            nn.Linear(state_dim + 3, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, state_dim)
        )
        
        self.teacher_forcing_ratio = 0.5
        
    def _generate_square_subsequent_mask(self, sz):
        """生成解码器的因果 mask"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(self, hist_state, hist_control, base_state, cfg_tensor, gt_pred=None):
        batch_size = hist_state.size(0)
        
        # 动态获取预测长度
        pred_len = base_state.size(1)
        
        # 将历史状态和控制信号拼接
        # hist_state: (batch, hist_len, state_dim)
        # hist_control: (batch, hist_len, control_dim)
        hist_input = torch.cat([hist_state, hist_control], dim=-1)  # (batch, hist_len, state_dim+control_dim)
        
        # 嵌入
        hist_emb = self.input_embedding(hist_input)  # (batch, hist_len, d_model)
        
        # 添加位置编码
        hist_emb = hist_emb + self.pos_encoder[:, :self.hist_len, :]
        
        # 编码器
        encoder_out = self.encoder(hist_emb)  # (batch, hist_len, d_model)
        
        # 解码器：使用 base_state 作为初始输入
        cfg = cfg_tensor[:, :3]  # (batch, 3)
        
        # 初始化解码器输入（使用 base_state 的第一个时间步）
        decoder_input = base_state[:, 0, :].unsqueeze(1)  # (batch, 1, state_dim)
        decoder_emb = self.decoder_embedding(decoder_input)  # (batch, 1, d_model)
        decoder_emb = decoder_emb + self.pos_encoder[:, self.hist_len:self.hist_len+1, :]
        
        delta_stack = []
        corrected_stack = []
        
        # 自回归解码
        for t in range(pred_len):
            # 解码器 forward
            if t == 0:
                dec_out = self.decoder(decoder_emb, encoder_out)  # (batch, 1, d_model)
            else:
                dec_out = self.decoder(decoder_emb, encoder_out)
            
            # 预测 delta
            delta_pred = self.output_proj(dec_out).squeeze(1)  # (batch, state_dim)
            
            # 使用残差 MLP refinement
            if t == 0:
                current_state = base_state[:, 0, :]
            else:
                current_state = corrected_stack[-1]
            
            refined_delta = self.residual_mlp(torch.cat([current_state, cfg], dim=-1))
            
            # 计算校正后的状态
            corrected = current_state + refined_delta
            
            delta_stack.append(refined_delta)
            corrected_stack.append(corrected)
            
            # 更新解码器输入（用于下一步）
            if t < pred_len - 1:
                # Teacher forcing
                if self.training and gt_pred is not None and torch.rand(1).item() < self.teacher_forcing_ratio:
                    target_state = gt_pred[:, t+1, :]
                else:
                    target_state = corrected.detach()
                
                # 使用 base_state 的 residual
                base_step = base_state[:, t+1, :] if t+1 < base_state.size(1) else base_state[:, 0, :]
                next_decoder_input = base_step + (target_state - base_step)
                
                # 重新嵌入
                decoder_emb = self.decoder_embedding(next_decoder_input.unsqueeze(1))
                decoder_emb = decoder_emb + self.pos_encoder[:, self.hist_len+t+1:self.hist_len+t+2, :]
        
        delta_stack = torch.stack(delta_stack, dim=1)  # (batch, pred_len, state_dim)
        corrected_stack = torch.stack(corrected_stack, dim=1)  # (batch, pred_len, state_dim)
        
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
