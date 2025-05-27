import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from models import AutoEncoder

# ==== 字体设置（避免中文+下标警告） ====
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==== 读取所有 SpO2 数据 ====
def load_all_spo2_data(data_dir="data"):
    all_sequences = []
    file_map = {}  # 映射每段属于哪个文件
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            path = os.path.join(data_dir, filename)
            values = []
            try:
                with open(path, 'r') as f:
                    for line in f:
                        try:
                            values.append(float(line.strip()))
                        except ValueError:
                            continue
                if len(values) >= 60:
                    all_sequences.append(values)
                    file_map[filename] = values
                    print(f"✅ 读取 {filename}（{len(values)} 条）")
            except Exception as e:
                print(f"❌ 读取失败 {filename}: {e}")
    return all_sequences, file_map

# ==== 切片函数 ====
def slice_sequences(sequences, window_size=60):
    segments = []
    segment_origin = []
    for idx, seq in enumerate(sequences):
        t = torch.tensor(seq, dtype=torch.float32)
        for i in range(len(t) - window_size):
            segments.append(t[i:i + window_size])
            segment_origin.append((idx, i))  # 第 idx 个序列的第 i 个窗口
    return torch.stack(segments), segment_origin

# ==== 加载数据 ====
all_sequences, file_map = load_all_spo2_data("data")
segments_tensor, segment_origin = slice_sequences(all_sequences, window_size=60)
print("📊 总样本数：", len(segments_tensor))

# ==== 创建数据集和加载器 ====
dataset = TensorDataset(segments_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ==== 初始化模型 ====
model = AutoEncoder(input_dim=60)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==== 训练模型 ====
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    for (batch,) in dataloader:
        output = model(batch)
        loss = criterion(output, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"📘 Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.6f}")

# ==== 异常检测 ====
model.eval()
reconstruction_errors = []
with torch.no_grad():
    for (batch,) in dataloader:
        output = model(batch)
        loss = torch.mean((output - batch) ** 2, dim=1)
        reconstruction_errors.extend(loss.numpy())

reconstruction_errors = np.array(reconstruction_errors)
threshold = np.percentile(reconstruction_errors, 95)
anomaly_indices = np.where(reconstruction_errors > threshold)[0]
print(f"🚨 异常阈值: {threshold}")
print(f"❗ 异常片段数量: {len(anomaly_indices)} / {len(reconstruction_errors)}")

# ==== 保存重构误差图 ====
plt.figure(figsize=(10, 4))
plt.plot(reconstruction_errors, label="重构误差")
plt.axhline(threshold, color='r', linestyle='--', label="阈值")
plt.scatter(anomaly_indices, reconstruction_errors[anomaly_indices], color='red', s=10, label="异常")
plt.title("SpO2 异常检测 - 重构误差分布")
plt.xlabel("样本序号")
plt.ylabel("误差")
plt.legend()
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/anomaly_detection.png")
print("✅ 重构误差图已保存：output/anomaly_detection.png")

# ==== 可视化：标注原始波形中的异常片段（以第一个文件为例） ====
# 找出第一个文件名
first_file = list(file_map.keys())[0]
original_data = file_map[first_file]
original_tensor = torch.tensor(original_data)

# 找出该文件中哪些片段被判定为异常
file_indices = [i for i, (f_idx, start) in enumerate(segment_origin) if f_idx == 0]
file_anomalies = [i for i in file_indices if i in anomaly_indices]
highlight_ranges = [segment_origin[i][1] for i in file_anomalies]

# 绘制原始波形 + 异常区域
plt.figure(figsize=(12, 4))
plt.plot(original_tensor, label="SpO2 原始波形")

for start in highlight_ranges:
    plt.axvspan(start, start + 60, color='red', alpha=0.3)

plt.title(f"{first_file} 血氧波形中异常片段标注")
plt.xlabel("时间（秒）")
plt.ylabel("SpO2 (%)")
plt.legend()
plt.tight_layout()
plt.savefig("output/spo2_with_anomalies.png")
print("✅ 标注图已保存：output/spo2_with_anomalies.png")

import pandas as pd

# === 创建保存图像目录 ===
plot_dir = os.path.join("output", "anomaly_plots")
os.makedirs(plot_dir, exist_ok=True)

stats = []

# 遍历每个文件
for file_idx, filename in enumerate(file_map.keys()):
    original_data = file_map[filename]
    original_tensor = torch.tensor(original_data)

    # 找出该文件所有片段在 segment_origin 中的位置
    file_segment_ids = [i for i, (fidx, _) in enumerate(segment_origin) if fidx == file_idx]
    file_anomaly_ids = [i for i in file_segment_ids if i in anomaly_indices]

    # 标注这些片段在原始信号中的位置
    highlight_ranges = [segment_origin[i][1] for i in file_anomaly_ids]

    # === 绘图 ===
    plt.figure(figsize=(12, 4))
    plt.plot(original_tensor, label="SpO2 原始波形")

    for start in highlight_ranges:
        plt.axvspan(start, start + 60, color='red', alpha=0.3)

    plt.title(f"{filename} 血氧异常片段标注")
    plt.xlabel("时间（秒）")
    plt.ylabel("SpO2 (%)")
    plt.legend()
    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(plot_dir, f"{filename.replace('.txt', '')}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 已保存图：{save_path}")

    # 统计信息
    stats.append({
        "文件名": filename,
        "总片段数": len(file_segment_ids),
        "异常片段数": len(file_anomaly_ids),
        "异常率(%)": round(100 * len(file_anomaly_ids) / len(file_segment_ids), 2)
    })

# === 保存异常统计表格 ===
df = pd.DataFrame(stats)
df.to_csv("output/anomaly_stats.csv", index=False, encoding="utf-8-sig")
print("📊 异常统计表已保存为：output/anomaly_stats.csv")

# ==== 导出为 ONNX 模型 ====
onnx_path = "output/autoencoder_spo2.onnx"
dummy_input = torch.randn(1, 60)  # 输入形状必须与训练时一致

torch.onnx.export(
    model,                        # 训练好的 PyTorch 模型
    dummy_input,                  # 模拟输入
    onnx_path,                    # 输出路径
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=11,
    verbose=False
)

print(f"✅ ONNX 模型已导出：{onnx_path}")

torch.save(model.state_dict(), "output/spo2_autoencoder.pth")

