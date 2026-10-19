# -*- coding: utf-8 -*-
"""
数据集离线压缩 - 一次性运行
将原始高分辨率图片统一缩放到指定尺寸，体积可减少 90%+
"""
import os
import cv2
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==================== 配置 ====================
SRC_DIR = 'dataset'  # 原始数据集目录
DST_DIR = 'dataset_small'  # 压缩后目录
TARGET_SIZE = 200  # 短边缩放到200px（训练时会crop到160）
JPEG_QUALITY = 85  # JPEG质量
NUM_WORKERS = 8  # i7-13620H 有10核，用8个进程


def compress_one(args):
    """压缩单张图片（保持宽高比）"""
    src_path, dst_path = args
    try:
        img = cv2.imread(src_path)
        if img is None:
            return False, src_path, "读取失败"

        h, w = img.shape[:2]

        # 短边缩到 TARGET_SIZE，长边等比（避免变形）
        scale = TARGET_SIZE / min(h, w)
        if scale < 1.0:  # 只缩小，不放大
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 保存为JPEG
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return True, src_path, None
    except Exception as e:
        return False, src_path, str(e)


def main():
    print(f"{'=' * 60}")
    print(f"数据集压缩工具")
    print(f"{'=' * 60}")
    print(f"源目录:     {SRC_DIR}")
    print(f"目标目录:   {DST_DIR}")
    print(f"短边尺寸:   {TARGET_SIZE}px")
    print(f"JPEG质量:   {JPEG_QUALITY}")
    print(f"并行进程:   {NUM_WORKERS}")
    print(f"{'=' * 60}\n")

    if not os.path.exists(SRC_DIR):
        print(f"❌ 源目录不存在: {SRC_DIR}")
        return

    # 收集所有图片
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG', '*.bmp']
    src_paths = []
    for ext in extensions:
        src_paths.extend(glob.glob(f'{SRC_DIR}/**/{ext}', recursive=True))

    if not src_paths:
        print(f"❌ 未找到图片")
        return

    print(f"找到 {len(src_paths)} 张图片")

    # 构建任务列表
    tasks = []
    for src in src_paths:
        rel_path = os.path.relpath(src, SRC_DIR)
        rel_path = os.path.splitext(rel_path)[0] + '.jpg'
        dst = os.path.join(DST_DIR, rel_path)
        tasks.append((src, dst))

    # 统计原始大小
    orig_size = sum(os.path.getsize(s) for s in src_paths) / (1024 ** 2)
    print(f"原始总大小: {orig_size:.1f} MB\n")

    # 并行压缩
    success, failed = 0, 0
    failed_list = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(compress_one, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="压缩进度"):
            ok, path, err = fut.result()
            if ok:
                success += 1
            else:
                failed += 1
                failed_list.append((path, err))

    # 统计压缩后大小
    new_size = 0
    for dp, _, fs in os.walk(DST_DIR):
        for f in fs:
            new_size += os.path.getsize(os.path.join(dp, f))
    new_size = new_size / (1024 ** 2)

    print(f"\n{'=' * 60}")
    print(f"✓ 成功: {success}    失败: {failed}")
    print(f"原始大小:   {orig_size:.1f} MB")
    print(f"压缩后大小: {new_size:.1f} MB")
    if orig_size > 0:
        print(f"压缩率:     {new_size / orig_size * 100:.1f}% (节省 {100 - new_size / orig_size * 100:.1f}%)")
    print(f"新数据集:   {DST_DIR}")
    print(f"{'=' * 60}")

    if failed_list[:5]:
        print("\n失败样例:")
        for p, e in failed_list[:5]:
            print(f"  {p}: {e}")


if __name__ == '__main__':
    main()