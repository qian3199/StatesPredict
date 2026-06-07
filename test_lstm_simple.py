import torch
import torch.nn as nn
import numpy as np

print("=== LSTM 简单测试 ===")

# 创建一个简单的LSTM模型
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=4):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 创建模型
model = SimpleLSTM()
print(f"模型结构:\n{model}")

# 生成模拟数据
batch_size = 8
seq_len = 15
input_size = 6

x = torch.randn(batch_size, seq_len, input_size)
print(f"\n输入形状: {x.shape}")

# 前向传播
output = model(x)
print(f"输出形状: {output.shape}")

# 测试损失函数
target = torch.randn(batch_size, 4)
criterion = nn.MSELoss()
loss = criterion(output, target)
print(f"损失值: {loss.item():.6f}")

# 测试反向传播
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("\n反向传播测试通过!")

print("\n=== LSTM 测试完成 ===")
