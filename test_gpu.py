import torch
import sys

print("=" * 60)
print("GPU兼容性测试")
print("=" * 60)

print(f"\nPyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备数量: {torch.cuda.device_count()}")
    print(f"当前GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU设备能力: {torch.cuda.get_device_capability(0)}")
    
    device = torch.device('cuda')
    print(f"\n测试GPU计算...")
    
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.matmul(x, y)
    print(f"✓ GPU矩阵运算成功! 结果shape: {z.shape}")
    
    from model import DyTR_LSTM
    model = DyTR_LSTM().to(device)
    print(f"✓ 模型成功移到GPU!")
    
    hist_state = torch.randn(32, 15, 4).to(device)
    hist_control = torch.randn(32, 15, 2).to(device)
    base_state = torch.randn(32, 10, 4).to(device)
    cfg_tensor = torch.FloatTensor([15, 10, 1]).repeat(32, 1).to(device)
    
    delta, corrected = model(hist_state, hist_control, base_state, cfg_tensor)
    print(f"✓ 前向传播成功! Delta shape: {delta.shape}, Corrected shape: {corrected.shape}")
    
    print("\n" + "=" * 60)
    print("所有GPU测试通过! ✓")
    print("=" * 60)
else:
    print("\n警告: CUDA不可用，将使用CPU")
    print("在具有NVIDIA GPU的系统上，CUDA应该可用")
    print("\n提示:")
    print("1. 确认已安装支持CUDA的PyTorch版本:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("2. 确认已安装NVIDIA驱动程序")
    print("3. 确认GPU未被其他程序占用")
    sys.exit(1)
