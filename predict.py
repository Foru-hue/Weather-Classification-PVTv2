# -*- coding: utf-8 -*-
"""
单张图片预测
用法1（推荐，零配置）：
    python predict.py                     # 自动从 test_images/ 或 dataset_small/ 随机挑
用法2（指定文件夹批量预测）：
    python predict.py --folder test_images
用法3（指定单张图片）：
    python predict.py --image path/to/img.jpg
"""
import os
import sys
import time
import glob
import random
import pickle
import argparse
import numpy as np
import cv2

# ⭐ 全局配置中文字体
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import paddle
from train import build_model, IMG_SIZE


LABEL_CN = {
    'dew': '露水', 'fogsmog': '雾霾', 'frost': '霜冻', 'glaze': '雨凇',
    'hail': '冰雹', 'lightning': '闪电', 'rain': '雨', 'rainbow': '彩虹',
    'rime': '雾凇', 'sandstorm': '沙尘暴', 'snow': '雪',
}


def preprocess_image(img_path, img_size=IMG_SIZE):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    s = min(h, w)
    y1 = (h - s) // 2
    x1 = (w - s) // 2
    img = img[y1:y1+s, x1:x1+s]
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype('float32') / 255.0
    img = img.transpose([2, 0, 1])
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img = (img - mean) / std
    return img, cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)


def predict_one(model, enc, img_path):
    img_tensor, img_display = preprocess_image(img_path)
    img_tensor = paddle.to_tensor(img_tensor[None, ...], dtype='float32')
    t0 = time.time()
    with paddle.no_grad():
        logits = model(img_tensor)
        probs = paddle.nn.functional.softmax(logits, axis=1).numpy()[0]
    infer_ms = (time.time() - t0) * 1000

    pred_idx = int(np.argmax(probs))
    pred_label = enc.classes_[pred_idx]
    top5_idx = np.argsort(probs)[::-1][:5]
    top5 = [(enc.classes_[i], float(probs[i])) for i in top5_idx]

    return {
        'label': pred_label,
        'label_cn': LABEL_CN.get(pred_label, pred_label),
        'confidence': float(probs[pred_idx]),
        'top5': top5,
        'infer_time_ms': infer_ms,
        'image': img_display,
        'img_path': img_path,
    }


def find_image(args):
    """按优先级找图片路径"""
    # 1. 命令行指定单张
    if args.image and os.path.exists(args.image):
        return args.image

    # 2. 命令行指定文件夹
    if args.folder and os.path.isdir(args.folder):
        imgs = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            imgs.extend(glob.glob(f'{args.folder}/**/{ext}', recursive=True))
        if imgs:
            chosen = random.choice(imgs)
            print(f"📂 从 {args.folder} 随机选择: {chosen}")
            return chosen

    # 3. 默认 test_images 文件夹
    if os.path.isdir('test_images'):
        imgs = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            imgs.extend(glob.glob(f'test_images/**/{ext}', recursive=True))
        if imgs:
            chosen = random.choice(imgs)
            print(f"📂 从 test_images/ 随机选择: {chosen}")
            return chosen

    # 4. 退到 dataset_small 随机
    if os.path.isdir('dataset_small'):
        imgs = glob.glob('dataset_small/**/*.jpg', recursive=True)
        if imgs:
            chosen = random.choice(imgs)
            print(f"📂 从 dataset_small/ 随机选择: {chosen}")
            return chosen

    return None


def visualize(result, save_path='predict_result.png', show=True):
    """可视化结果 - 大图 + Top5 条形图"""
    fig = plt.figure(figsize=(14, 6))

    # 左：图片
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(result['image'])
    ax1.set_title(
        f"预测: {result['label_cn']}   置信度: {result['confidence']*100:.1f}%",
        fontsize=16, pad=12)
    ax1.axis('off')

    # 右：Top5 条形图
    ax2 = plt.subplot(1, 2, 2)
    labels = [f"{LABEL_CN.get(lb, lb)}\n({lb})" for lb, _ in result['top5']]
    values = [p * 100 for _, p in result['top5']]
    colors = ['#2ecc71' if i == 0 else '#95a5a6' for i in range(len(values))]
    bars = ax2.barh(range(len(values))[::-1], values, color=colors)
    ax2.set_yticks(range(len(values))[::-1])
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.set_xlabel('置信度 (%)', fontsize=12)
    ax2.set_title('Top-5 预测', fontsize=14)
    ax2.set_xlim(0, 105)
    for bar, val in zip(bars, values):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}%", va='center', fontsize=10)

    plt.suptitle(f"推理耗时: {result['infer_time_ms']:.2f} ms  |  "
                 f"图片: {os.path.basename(result['img_path'])}",
                 fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\n✓ 结果图已保存: {save_path}")



def print_result(result):
    print(f"\n{'='*50}")
    print(f"🎯 预测结果: {result['label_cn']}  ({result['label']})")
    print(f"📊 置信度:   {result['confidence']*100:.2f}%")
    print(f"⏱ 推理耗时: {result['infer_time_ms']:.2f} ms")
    print(f"{'='*50}")
    print(f"\nTop-5 预测:")
    for i, (lb, pr) in enumerate(result['top5'], 1):
        cn = LABEL_CN.get(lb, lb)
        bar = '█' * int(pr * 30)
        print(f"  {i}. {cn:<8}({lb:<10}) {pr*100:6.2f}%  {bar}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="天气图片预测")
    parser.add_argument('--image', type=str, default=None,
                        help='指定单张图片路径')
    parser.add_argument('--folder', type=str, default=None,
                        help='指定文件夹，随机抽一张')
    parser.add_argument('--no-show', action='store_true',
                        help='不弹窗，只保存结果图')
    args = parser.parse_args()

    if paddle.device.is_compiled_with_cuda():
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')

    # 加载 LabelEncoder
    if not os.path.exists('label_encoder.pkl'):
        print("❌ 找不到 label_encoder.pkl，请先运行 天气识别.py 训练一次")
        return
    with open('label_encoder.pkl', 'rb') as f:
        enc = pickle.load(f)

    # 加载模型
    if not os.path.exists('best_model.pdparams'):
        print("❌ 找不到 best_model.pdparams，请先训练")
        return
    model = build_model(num_classes=len(enc.classes_))
    model.set_state_dict(paddle.load('best_model.pdparams'))
    model.eval()
    print(f"✓ 模型加载成功，类别数: {len(enc.classes_)}\n")

    # 找图片
    img_path = find_image(args)
    if img_path is None:
        print("❌ 找不到可预测的图片")
        print("   请把图片放到 test_images/ 目录，或用 --image 指定")
        return

    if not os.path.exists(img_path):
        print(f"❌ 图片不存在: {img_path}")
        return

    # 预测
    print(f"📷 图片路径: {img_path}")
    result = predict_one(model, enc, img_path)

    # 打印结果
    print_result(result)

    # 可视化
    visualize(result, show=not args.no_show)


if __name__ == '__main__':
    main()