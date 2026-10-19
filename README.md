# 🌤️ 基于 PVT v2 的十一类天气图像识别系统

本项目是一个基于前沿视觉 Transformer 网络 **PVT v2** 和 **PaddlePaddle** 框架的 11 类天气图像分类系统。涵盖了从数据清洗与离线压缩、模型训练与调优、到 Web 端可视化部署的完整全流程。

## 📊 项目成果

- **模型精度**：Accuracy **97.94%** / Precision **98.15%** 
- **推理速度**：单张图像推理仅需 **~7ms** (在 RTX 4050 环境下测试)
- **数据压缩**：编写离线压缩脚本，将原始数据集从 1.2GB 缩减至 92MB（**压缩率 92.4%**），极大提升 I/O 读取效率
- **覆盖类别**：露水 (dew), 雾霾 (fogsmog), 霜冻 (frost), 雨凇 (glaze), 冰雹 (hail), 闪电 (lightning), 雨 (rain), 彩虹 (rainbow), 雾凇 (rime), 沙尘暴 (sandstorm), 雪 (snow)

## ✨ 核心特性

1. **前沿模型**：采用 PVT v2 (B2-Linear) 架构，兼具 Transformer 的全局感受野与 CNN 的轻量化优势。
2. **丰富的特征增强**：针对天气图像特点设计了 7 种数据增强策略（缩放裁剪、水平/垂直翻转、旋转、色彩抖动、高斯模糊模拟雾天、Cutout 遮挡）。
3. **显存优化**：在仅 6GB 显存的 RTX 4050 上，引入 **AMP（自动混合精度）** 与 **余弦退火学习率 (CosineAnnealing)**，确保大模型在有限显存下稳定高效收敛。
4. **Web 可视化部署**：集成 Gradio 框架，支持图片拖拽上传、实时分类与置信度动态条形图展示。

## 🖼️ 界面展示

![训练曲线](training_curves.png)
![混淆矩阵](confusion_matrix.png)
![Web演示](web_demo.png)

## 📥 权重下载

由于 GitHub 文件大小限制，预训练模型与训练好的最佳权重请通过以下方式下载：
- **百度网盘链接**: [https://pan.baidu.com/s/1YkyRA1ZlrNZOvM0v6A9PtQ?pwd=adp2](https://pan.baidu.com/s/1YkyRA1ZlrNZOvM0v6A9PtQ?pwd=adp2) 
- **提取码**: `adp2`

> **注意**：下载后请将 `pvt_v2_b2_li.pdparams`（预训练模型）和 `best_model.pdparams`（最佳权重）放置在项目根目录下（与 `train.py` 同级）。

## 🚀 快速开始

### 1. 环境安装
```bash
pip install -r requirements.txt
```

### 2. 数据集准备与压缩
本项目使用的天气数据集来源于 Kaggle 开源的 [Weather Image Recognition](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset) 。
- 数据集包含 11 类天气，共 13724 张图片。
- 请下载并解压后，将其重命名为 `dataset`，放置在项目根目录下。
- 然后运行 `python compress_dataset.py` 自动进行数据压缩（将生成 92MB 的 `dataset_small` 目录）。

### 3. 模型训练与评估
```bash
python train.py
```
(会自动保存最佳模型 best_model.pdparams、混淆矩阵图片以及训练曲线)

### 4. 单图预测测试
```bash
python predict.py --image path/to/your/image.jpg
```

### 5. 启动 Web 可视化界面
```bash
python app.py
```
启动后，在浏览器中打开 http://127.0.0.1:7860 即可体验实时天气分类。
