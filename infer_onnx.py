import onnxruntime as ort
import numpy as np
import torch

# === 加载你导出的 ONNX 模型 ===
onnx_path = "output/autoencoder_spo2.onnx"
session = ort.InferenceSession(onnx_path)

# === 构造一个示例 SpO2 片段（或从你数据中选一段）
# 这里随机构造一个 [60] 长度的 SpO₂ 片段
# 你也可以用真实数据替换
example_segment = np.random.normal(loc=98, scale=0.5, size=(60,)).astype(np.float32)
input_data = example_segment.reshape(1, -1)  # ONNX 模型要求输入为 [batch_size, 60]

# === 运行推理（预测重构结果）
outputs = session.run(None, {"input": input_data})
reconstructed = outputs[0]

# === 计算重构误差（均方误差）
mse = np.mean((input_data - reconstructed) ** 2)
print("✅ 重构误差（MSE）:", mse)

# === 判断是否为异常（可设置你保存时的 threshold）
threshold = 0.8  # 示例值，请使用你真实训练时的那个值
if mse > threshold:
    print("🚨 判断结果：异常")
else:
    print("✅ 判断结果：正常")
