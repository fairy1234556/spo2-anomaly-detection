import os
import numpy as np
import torch
import torch.nn as nn
from models import AutoEncoder

# === 参数设置 ===
txt_file = "data/010217A.txt"  # 替换成你自己的文件路径
window_size = 60
threshold = 0.3  # MSE 重构误差阈值

# === 加载模型 ===
model = AutoEncoder()
model.load_state_dict(torch.load("output/spo2_autoencoder.pth"))
model.eval()

loss_fn = nn.MSELoss()

# === 读取 SpO2 数据 ===
spo2_values = []
with open(txt_file, 'r') as f:
    for line in f:
        try:
            val = float(line.strip())
            spo2_values.append(val)
        except:
            continue
data = np.array(spo2_values, dtype=np.float32)

# === 切片为窗口 ===
segments = []
for i in range(len(data) - window_size + 1):
    segment = data[i:i + window_size]
    segments.append(segment)
segments = np.stack(segments)  # [N, 60]
inputs = torch.tensor(segments)

# === 推理并计算 Loss ===
with torch.no_grad():
    outputs = model(inputs)
    losses = torch.mean((outputs - inputs) ** 2, dim=1)

# === 输出前几项 Loss
for i in range(min(20, len(losses))):
    status = "⚠️ 异常" if losses[i] > threshold else "✅ 正常"
    print(f"[{i}] Loss: {losses[i]:.6f} {status}")

# === 统计异常窗口数量
anomaly_count = (losses > threshold).sum().item()
print(f"\n📊 异常窗口数: {anomaly_count} / {len(losses)}")
