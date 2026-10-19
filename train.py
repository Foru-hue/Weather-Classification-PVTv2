# -*- coding: utf-8 -*-
"""
PVT v2 天气分类 - 完整版
新增：每轮耗时 + 总耗时 + 预测耗时 + 混淆矩阵 + 保存 LabelEncoder
"""
import os
import cv2
import glob
import time
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from tqdm import tqdm
import pickle

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset

# ==================== ⚙️ 配置 ====================
IMG_SIZE = 160
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 20
LR = 1e-4
USE_AMP = True

DATA_ROOT = 'dataset_small'
PRETRAINED_WEIGHTS = 'pvt_v2_b2_li.pdparams'

if not os.path.exists(DATA_ROOT):
    print(f"⚠️ 找不到数据集 {DATA_ROOT}，请先运行 compress_dataset.py")


# ==================== 时间工具 ====================
def format_time(seconds):
    """秒 → 可读格式"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def setup_environment():
    print(f"{'='*60}")
    print(f"PaddlePaddle 版本: {paddle.__version__}")
    print(f"CUDA 编译支持: {paddle.device.is_compiled_with_cuda()}")
    if paddle.device.is_compiled_with_cuda():
        paddle.set_device('gpu')
        try:
            gpu_info = paddle.device.cuda.get_device_properties(0)
            print(f"GPU: {gpu_info.name}")
            print(f"显存: {gpu_info.total_memory / 1024**3:.1f} GB")
        except Exception:
            pass
        print(f"✅ 使用 GPU 训练")
    else:
        paddle.set_device('cpu')
        print(f"⚠️ 使用 CPU 训练")
    print(f"{'='*60}\n")


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); paddle.seed(seed)


# ==================== 数据 ====================
def prepare_data(path):
    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    paths = []
    for e in exts:
        paths.extend(glob.glob(f'{path}/**/{e}', recursive=True))
    print(f"找到 {len(paths)} 张图片")
    valid = []
    for p in tqdm(paths, desc="校验图片"):
        img = cv2.imread(p)
        if isinstance(img, np.ndarray):
            valid.append(p)
    print(f"有效: {len(valid)}")
    labels = [os.path.basename(os.path.dirname(p)) for p in valid]
    df = pd.DataFrame({'File_Path': valid, 'Labels': labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\n类别分布:\n{df['Labels'].value_counts()}")
    return df


def plot_label_distribution(data, save_path='label_dist.png'):
    counts = data.Labels.value_counts()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=counts.index, y=counts.values)
    plt.axhline(y=counts.mean(), lw=3, color="#346c9a")
    plt.xlabel('Labels'); plt.ylabel('Count')
    plt.xticks(rotation=50)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"类别分布图: {save_path}")


class WeatherDataset(Dataset):
    def __init__(self, data, enc, rate=0.7, mode='train', img_size=IMG_SIZE):
        super().__init__()
        self.data = data.copy()
        self.data['Labels'] = enc.transform(self.data['Labels'])
        self.img_size = img_size
        self.mode = mode
        n = len(self.data)
        split = int(n * rate)
        if mode == 'train':
            self.data = self.data.iloc[:split].reset_index(drop=True)
        else:
            self.data = self.data.iloc[split:].reset_index(drop=True)

    def _augment(self, img):
        h, w = img.shape[:2]
        if random.random() < 0.5:
            scale = random.uniform(0.75, 1.0)
            nh, nw = int(h * scale), int(w * scale)
            r = cv2.resize(img, (nw, nh))
            canvas = np.zeros((h, w, 3), dtype=img.dtype)
            y1 = random.randint(0, max(0, h - nh))
            x1 = random.randint(0, max(0, w - nw))
            canvas[y1:y1+nh, x1:x1+nw] = r
            img = canvas
        if random.random() < 0.5: img = np.fliplr(img).copy()
        if random.random() < 0.3: img = np.flipud(img).copy()
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-20, 20)
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        if random.random() < 0.3:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        if random.random() < 0.3:
            for _ in range(random.randint(1, 3)):
                ch = random.randint(10, h // 4); cw = random.randint(10, w // 4)
                y = random.randint(0, h - ch); x = random.randint(0, w - cw)
                img[y:y+ch, x:x+cw] = np.random.randint(0, 255, (ch, cw, 3), dtype=np.uint8)
        return img

    def __getitem__(self, idx):
        path, label = self.data.iloc[idx]
        img = cv2.imread(path)
        if not isinstance(img, np.ndarray):
            img = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            s = min(h, w)
            y1 = (h - s) // 2; x1 = (w - s) // 2
            img = img[y1:y1+s, x1:x1+s]
            img = cv2.resize(img, (self.img_size, self.img_size))
        if self.mode == 'train':
            img = self._augment(img)
        img = img.astype('float32') / 255.0
        img = img.transpose([2, 0, 1])
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img = (img - mean) / std
        return img, int(label)

    def __len__(self):
        return len(self.data)


# ==================== PVT v2 模型 ====================
def to_2tuple(x): return tuple([x] * 2)

def swapdim(x, d1, d2):
    a = list(range(len(x.shape))); a[d1], a[d2] = a[d2], a[d1]
    return x.transpose(a)


def drop_path(x, dp=0., tr=False):
    if dp == 0. or not tr: return x
    keep = 1 - dp
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rt = paddle.to_tensor(keep) + paddle.rand(shape)
    rt = paddle.floor(rt)
    return x.divide(keep) * rt


class DropPath(nn.Layer):
    def __init__(self, dp=None): super().__init__(); self.dp = dp
    def forward(self, x): return drop_path(x, self.dp, self.training)


class Identity(nn.Layer):
    def forward(self, x): return x


class DWConv(nn.Layer):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2D(dim, dim, 3, 1, 1, bias_attr=True, groups=dim)
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = swapdim(x, 1, 2).reshape([B, C, H, W])
        x = self.dwconv(x).flatten(2)
        return swapdim(x, 1, 2)


class Mlp(nn.Layer):
    def __init__(self, in_f, hid_f=None, out_f=None, act=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_f = out_f or in_f
        hid_f = hid_f or in_f
        self.fc1 = nn.Linear(in_f, hid_f)
        self.dwconv = DWConv(hid_f)
        self.act = act()
        self.fc2 = nn.Linear(hid_f, out_f)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if linear: self.relu = nn.ReLU()
    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear: x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x); x = self.drop(x)
        x = self.fc2(x); return self.drop(x)


class Attention(nn.Layer):
    def __init__(self, dim, nh=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        self.dim = dim; self.nh = nh
        hd = dim // nh
        self.scale = qk_scale or hd ** -0.5
        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.linear = linear; self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2D(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2D(7)
            self.sr = nn.Conv2D(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape([B, N, self.nh, C // self.nh]).transpose([0, 2, 1, 3])
        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.transpose([0, 2, 1]).reshape([B, C, H, W])
                x_ = self.sr(x_).reshape([B, C, -1]).transpose([0, 2, 1])
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape([B, -1, 2, self.nh, C // self.nh]).transpose([2, 0, 3, 1, 4])
            else:
                kv = self.kv(x).reshape([B, -1, 2, self.nh, C // self.nh]).transpose([2, 0, 3, 1, 4])
        else:
            x_ = x.transpose([0, 2, 1]).reshape([B, C, H, W])
            x_ = self.sr(self.pool(x_)).reshape([B, C, -1]).transpose([0, 2, 1])
            x_ = self.act(self.norm(x_))
            kv = self.kv(x_).reshape([B, -1, 2, self.nh, C // self.nh]).transpose([2, 0, 3, 1, 4])
        k, v = kv[0], kv[1]
        attn = F.softmax((q @ swapdim(k, -2, -1)) * self.scale, axis=-1)
        attn = self.attn_drop(attn)
        x = swapdim((attn @ v), 1, 2).reshape([B, N, C])
        return self.proj_drop(self.proj(x))


class Block(nn.Layer):
    def __init__(self, dim, nh, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act=nn.GELU,
                 norm=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm(dim)
        self.attn = Attention(dim, nh=nh, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop,
                              sr_ratio=sr_ratio, linear=linear)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm(dim)
        self.mlp = Mlp(in_f=dim, hid_f=int(dim * mlp_ratio), act=act, drop=drop, linear=linear)
    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Layer):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        ps = to_2tuple(patch_size)
        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=ps, stride=stride,
                              padding=(ps[0] // 2, ps[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = swapdim(x.flatten(2), 1, 2)
        return self.norm(x), H, W


class PyramidVisionTransformerV2(nn.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = num_stages
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(num_stages):
            pe = OverlapPatchEmbed(
                img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i])
            blk = nn.LayerList([Block(
                dim=embed_dims[i], nh=num_heads[i], mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm=norm_layer, sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            cur += depths[i]
            setattr(self, f"patch_embed{i + 1}", pe)
            setattr(self, f"block{i + 1}", blk)
            setattr(self, f"norm{i + 1}", norm_layer(embed_dims[i]))
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else Identity()

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(self.num_stages):
            x, H, W = getattr(self, f"patch_embed{i + 1}")(x)
            for blk in getattr(self, f"block{i + 1}"):
                x = blk(x, H, W)
            x = getattr(self, f"norm{i + 1}")(x)
            if i != self.num_stages - 1:
                x = x.reshape([B, H, W, -1]).transpose([0, 3, 1, 2])
        return x.mean(axis=1)

    def forward(self, x):
        return self.head(self.forward_features(x))


def build_model(num_classes=11):
    return PyramidVisionTransformerV2(
        img_size=IMG_SIZE, patch_size=4,
        embed_dims=[64, 128, 320, 512], num_classes=num_classes,
        num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], linear=True)


def load_pretrained(model, weight_path):
    if not os.path.exists(weight_path):
        print(f"⚠️ 未找到 {weight_path}，从头训练")
        return model
    print(f"加载预训练权重: {weight_path}")
    state = paddle.load(weight_path)
    model_dict = model.state_dict()
    loaded, skipped = 0, 0
    new_state = {}
    for k, v in state.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            new_state[k] = v; loaded += 1
        else: skipped += 1
    model_dict.update(new_state)
    model.set_state_dict(model_dict)
    print(f"✓ 加载 {loaded} 个参数，跳过 {skipped} 个\n")
    return model


# ==================== 训练 ====================
def train():
    setup_environment()
    set_seed(42)

    if not os.path.exists(DATA_ROOT):
        print(f"❌ 数据集路径不存在: {DATA_ROOT}")
        return None, None

    data = prepare_data(DATA_ROOT)
    plot_label_distribution(data)

    unique_labels = sorted(data['Labels'].unique())
    enc = preprocessing.LabelEncoder().fit(unique_labels)
    print(f"\n类别数: {len(unique_labels)}")
    print(f"标签映射: {dict(zip(enc.classes_, range(len(enc.classes_))))}\n")

    # ⭐ 保存 LabelEncoder，供 predict.py 和 app.py 使用
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(enc, f)
    print("✓ 已保存 label_encoder.pkl\n")

    train_ds = WeatherDataset(data, enc, rate=0.7, mode='train')
    val_ds = WeatherDataset(data, enc, rate=0.7, mode='test')
    print(f"训练集: {len(train_ds)}, 验证集: {len(val_ds)}\n")

    network = build_model(num_classes=len(unique_labels))
    network = load_pretrained(network, PRETRAINED_WEIGHTS)

    lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=LR, T_max=EPOCHS, eta_min=1e-6)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler, parameters=network.parameters(), weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    scaler = paddle.amp.GradScaler(enable=USE_AMP, init_loss_scaling=1024.0) if USE_AMP else None

    print(f"{'='*60}")
    print(f"训练配置: IMG={IMG_SIZE}, BS={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LR}, AMP={USE_AMP}")
    print(f"{'='*60}\n")

    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'epoch_time': []}
    total_start = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # ---------- 训练 ----------
        network.train()
        total_loss, correct, total = 0.0, 0, 0
        indices = np.random.permutation(len(train_ds))
        n_batches = (len(indices) + BATCH_SIZE - 1) // BATCH_SIZE

        pbar = tqdm(range(0, len(indices), BATCH_SIZE),
                    desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for i in pbar:
            batch_idx = indices[i:i+BATCH_SIZE]
            imgs = np.stack([train_ds[j][0] for j in batch_idx])
            lbls = np.array([train_ds[j][1] for j in batch_idx])
            imgs = paddle.to_tensor(imgs, dtype='float32')
            lbls = paddle.to_tensor(lbls, dtype='int64')
            if USE_AMP:
                with paddle.amp.auto_cast():
                    logits = network(imgs)
                    loss = criterion(logits, lbls)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = network(imgs)
                loss = criterion(logits, lbls)
                loss.backward()
                optimizer.step()
            optimizer.clear_grad()
            lr_scheduler.step()
            total_loss += loss.item()
            correct += (paddle.argmax(logits, axis=1) == lbls).sum().item()
            total += len(lbls)
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct/total:.4f}"})

        train_loss = total_loss / n_batches
        train_acc = correct / total

        # ---------- 验证 ----------
        network.eval()
        v_loss, v_correct, v_total, v_batches = 0.0, 0, 0, 0
        with paddle.no_grad():
            for i in range(0, len(val_ds), BATCH_SIZE):
                batch_idx = range(i, min(i+BATCH_SIZE, len(val_ds)))
                imgs = np.stack([val_ds[j][0] for j in batch_idx])
                lbls = np.array([val_ds[j][1] for j in batch_idx])
                imgs = paddle.to_tensor(imgs, dtype='float32')
                lbls = paddle.to_tensor(lbls, dtype='int64')
                logits = network(imgs)
                v_loss += criterion(logits, lbls).item()
                v_correct += (paddle.argmax(logits, axis=1) == lbls).sum().item()
                v_total += len(lbls); v_batches += 1
        val_loss = v_loss / max(1, v_batches)
        val_acc = v_correct / v_total

        epoch_time = time.time() - epoch_start
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)

        # ⭐ 打印每轮耗时 + 累计耗时
        elapsed = time.time() - total_start
        print(f"Epoch [{epoch+1:2d}/{EPOCHS}] "
              f"Train L:{train_loss:.4f} A:{train_acc:.4f} | "
              f"Val L:{val_loss:.4f} A:{val_acc:.4f} | "
              f"LR:{lr_scheduler.get_lr():.6f} | "
              f"⏱ 本轮:{format_time(epoch_time)} 累计:{format_time(elapsed)}")

        if val_acc > best_acc:
            best_acc = val_acc
            paddle.save(network.state_dict(), 'best_model.pdparams')
            print(f"  ✓ 保存最佳模型 (Val Acc: {best_acc:.4f})")
        paddle.device.cuda.empty_cache()

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"训练完成！总耗时: {format_time(total_time)}")
    print(f"平均每轮: {format_time(np.mean(history['epoch_time']))}")
    print(f"最佳验证准确率: {best_acc:.4f}")
    print(f"{'='*60}")

    plot_curves(history)
    return network, enc, data, total_time


def plot_curves(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'], label='Val', marker='s')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history['train_acc'], label='Train', marker='o')
    axes[1].plot(history['val_acc'], label='Val', marker='s')
    axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True)
    axes[2].bar(range(1, len(history['epoch_time']) + 1),
                history['epoch_time'], color='steelblue')
    axes[2].set_title('Time per Epoch (s)'); axes[2].grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("训练曲线: training_curves.png")


# ==================== 评估 + 混淆矩阵 ====================
def evaluate_and_cm(model_path, data, enc, save_cm='confusion_matrix.png'):
    print(f"\n{'='*60}\n模型评估 + 混淆矩阵\n{'='*60}")
    network = build_model(num_classes=len(enc.classes_))
    network.set_state_dict(paddle.load(model_path))
    network.eval()

    val_ds = WeatherDataset(data, enc, rate=0.7, mode='test')
    all_preds, all_labels = [], []

    # ⭐ 预测计时
    pred_start = time.time()
    with paddle.no_grad():
        for i in tqdm(range(0, len(val_ds), BATCH_SIZE), desc="评估"):
            batch_idx = range(i, min(i+BATCH_SIZE, len(val_ds)))
            imgs = np.stack([val_ds[j][0] for j in batch_idx])
            lbls = [val_ds[j][1] for j in batch_idx]
            imgs = paddle.to_tensor(imgs, dtype='float32')
            preds = paddle.argmax(network(imgs), axis=1).numpy().tolist()
            all_preds.extend(preds); all_labels.extend(lbls)
    pred_time = time.time() - pred_start

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"\n✅ Accuracy:  {acc:.4f}")
    print(f"✅ Precision: {prec:.4f}")
    print(f"⏱ 预测耗时: {format_time(pred_time)}  "
          f"(平均每张图 {pred_time/len(val_ds)*1000:.2f} ms)")

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=enc.classes_, yticklabels=enc.classes_,
                cbar_kws={'label': 'Accuracy'})
    plt.xlabel('Predicted', fontsize=13)
    plt.ylabel('True', fontsize=13)
    plt.title('Confusion Matrix (Normalized)', fontsize=15)
    plt.tight_layout()
    plt.savefig(save_cm, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"✓ 混淆矩阵: {save_cm}")

    print(f"\n{'类别':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print('-' * 50)
    for i, cls in enumerate(enc.classes_):
        p = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        r = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"{cls:<12} {p:<12.4f} {r:<12.4f} {f1:<12.4f}")
    print('=' * 50)

    return acc, prec, pred_time


# ==================== 主入口 ====================
if __name__ == '__main__':
    result = train()
    if result[0] is not None:
        _, enc, data, train_time = result
        evaluate_and_cm('best_model.pdparams', data, enc)
        print(f"\n🎯 端到端总耗时（含数据加载/训练/评估）: {format_time(train_time)}")