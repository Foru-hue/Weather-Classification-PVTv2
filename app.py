# -*- coding: utf-8 -*-
"""
Gradio Web 界面 - 天气图片分类可视化
运行: python app.py
然后浏览器打开: http://127.0.0.1:7860
"""
import huggingface_hub
if not hasattr(huggingface_hub, "HfFolder"):
    class _HfFolder:
        @staticmethod
        def get_token():
            return huggingface_hub.get_token()
    huggingface_hub.HfFolder = _HfFolder
import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import pickle
import time
import numpy as np
import cv2
import gradio as gr

import paddle

from train import build_model, IMG_SIZE


LABEL_CN = {
    'dew': '露水', 'fogsmog': '雾霾', 'frost': '霜冻', 'glaze': '雨凇',
    'hail': '冰雹', 'lightning': '闪电', 'rain': '雨', 'rainbow': '彩虹',
    'rime': '雾凇', 'sandstorm': '沙尘暴', 'snow': '雪',
}

# 全局模型
MODEL = None
ENC = None


def load_model():
    global MODEL, ENC
    if paddle.device.is_compiled_with_cuda():
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')

    if not os.path.exists('label_encoder.pkl'):
        raise FileNotFoundError("找不到 label_encoder.pkl，请先训练")
    if not os.path.exists('best_model.pdparams'):
        raise FileNotFoundError("找不到 best_model.pdparams，请先训练")

    with open('label_encoder.pkl', 'rb') as f:
        ENC = pickle.load(f)

    MODEL = build_model(num_classes=len(ENC.classes_))
    MODEL.set_state_dict(paddle.load('best_model.pdparams'))
    MODEL.eval()
    print(f"✓ 模型加载成功，支持 {len(ENC.classes_)} 个类别")


def preprocess(pil_img, img_size=IMG_SIZE):
    """PIL Image → 归一化 numpy array (CHW)"""
    img = np.array(pil_img)
    if img.shape[-1] == 4:
        img = img[..., :3]  # RGBA → RGB
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
    return img


def classify(img):
    """Gradio 回调函数：输入 PIL Image，输出 label->conf 字典"""
    if img is None:
        return {}, "请上传一张天气图片"

    t0 = time.time()
    img_tensor = preprocess(img)
    img_tensor = paddle.to_tensor(img_tensor[None, ...], dtype='float32')

    with paddle.no_grad():
        logits = MODEL(img_tensor)
        probs = paddle.nn.functional.softmax(logits, axis=1).numpy()[0]

    infer_ms = (time.time() - t0) * 1000

    # 全部类别返回中文标签 + 置信度
    results = {}
    for i, cls in enumerate(ENC.classes_):
        cn = LABEL_CN.get(cls, cls)
        results[f"{cn}({cls})"] = float(probs[i])

    # 结果摘要
    pred_idx = int(np.argmax(probs))
    pred_cn = LABEL_CN.get(ENC.classes_[pred_idx], ENC.classes_[pred_idx])
    pred_conf = float(probs[pred_idx]) * 100
    summary = (f"🎯 预测类别: {pred_cn}\n"
               f"📊 置信度: {pred_conf:.2f}%\n"
               f"⏱ 推理耗时: {infer_ms:.2f} ms")

    return results, summary


# ==================== Gradio 界面 ====================
def build_ui():
    with gr.Blocks(title="天气图片识别系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🌤️ 天气图片识别系统
        ### 基于 PVT v2 的前沿视觉 Transformer
        上传一张天气图片，系统会识别其类别（支持 11 类天气）
        """)

        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(type="pil", label="上传图片", height=350)
                btn = gr.Button("🔍 开始识别", variant="primary", size="lg")
                btn_clear = gr.Button("🗑️ 清空")

            with gr.Column(scale=1):
                label_output = gr.Label(num_top_classes=11, label="各类别置信度")
                text_output = gr.Textbox(label="识别结果", lines=4)

        # 示例图片
        gr.Examples(
            examples=[],  # 你可以把 dataset_small 里几个代表性图片的路径填进来
            inputs=img_input,
            label="示例（可选）"
        )

        gr.Markdown("""
        ---
        **支持的 11 个类别**:
        露水(dew) · 雾霾(fogsmog) · 霜冻(frost) · 雨凇(glaze) · 冰雹(hail) · 
        闪电(lightning) · 雨(rain) · 彩虹(rainbow) · 雾凇(rime) · 沙尘暴(sandstorm) · 雪(snow)

        **模型**: PVT v2-B2-Linear · 训练精度 97%+
        """)

        # 绑定事件
        btn.click(fn=classify, inputs=img_input, outputs=[label_output, text_output])
        btn_clear.click(fn=lambda: (None, {}, ""),
                        inputs=None, outputs=[img_input, label_output, text_output])

    return demo


if __name__ == '__main__':
    load_model()
    demo = build_ui()
    # 局域网访问加 server_name='0.0.0.0'
    demo.launch(server_name='127.0.0.1', server_port=7860, share=False)