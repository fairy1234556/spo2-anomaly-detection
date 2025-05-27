# SpO₂ Anomaly Detection with AutoEncoder

本项目基于 PyTorch 构建轻量级自编码器（AutoEncoder）模型，对血氧（SpO₂）时间序列数据进行异常检测，并支持导出为 ONNX 格式以便部署到嵌入式设备。

📦 项目结构：
```
spo2analyze/
├── data/                  # 存放原始 SpO₂ 数据的 .txt 文件
├── output/                # 保存图像、异常检测结果、ONNX 模型等
├── dataset.py             # 数据加载与窗口切片逻辑
├── models.py              # AutoEncoder 模型结构
├── main.py                # 主程序（训练 + 异常检测 + 可视化）
├── infer_onnx.py          # 使用 ONNX 模型进行推理
└── test_loss_ae.py        # 单独测试 AE 重构损失
```

---

## 🚀 快速开始

### 1️⃣ 安装依赖环境
确保你已安装 Python 3.8+ 和 pip，然后运行：
```bash
pip install -r requirements.txt
```

---

### 2️⃣ 准备数据
将你的 `.txt` 血氧数据文件放入 `data/` 文件夹下，每行一个 SpO₂ 数值，例如：
```
98.6
98.5
97.4
...
```

---

### 3️⃣ 运行主程序进行训练与异常检测
```bash
python main.py
```

程序会自动：
- 读取所有 txt 文件
- 切片为滑动窗口序列
- 使用 AutoEncoder 训练
- 计算重构误差进行异常判断
- 绘制异常标注图并保存到 `output/`

---

### 4️⃣ 导出 ONNX 模型
训练完成后，会自动导出：
```bash
output/autoencoder_spo2.onnx
```

---

### 5️⃣ 进行 ONNX 推理测试（可选）
```bash
python infer_onnx.py
```

---

### 6️⃣ 单独评估重构误差（Loss）
```bash
python test_loss_ae.py
```
可查看模型在每段数据上的重构误差。

---

## 📊 输出内容

- `output/anomaly_detection.png`：原始波形 + 异常标记
- `output/spo2_with_anomalies.png`：完整拼接图
- `output/anomaly_stats.csv`：各文件的异常片段统计
- `output/anomaly_plots/*.png`：每个文件的异常可视化
- `output/autoencoder_spo2.onnx`：ONNX 模型文件

---

## ✅ 项目特色

- 🧠 使用 AutoEncoder 自动学习正常模式，识别异常段
- 📉 可视化展示原始信号与异常点
- 🛠️ 轻量化模型设计，支持部署到 MCU / Edge AI
- 🧪 提供推理与 loss 测试脚本

---

## 📌 依赖包（requirements.txt）
```txt
numpy
matplotlib
torch
onnx
onnxruntime
```

---

## ✍️ 作者
本项目由 fairy123456 开发，用于血氧信号异常分析与边缘部署测试。如需合作或交流欢迎联系！
