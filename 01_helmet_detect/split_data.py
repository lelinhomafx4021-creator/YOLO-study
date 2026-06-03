# -*- coding: utf-8 -*-
"""
split_data.py - 数据集 8:2 随机划分脚本
自动从 datasets/safety_helmet/images/train 中随机抽取 20% 的图片及对应的标签，
剪切移动到 val/images 和 val/labels 目录下，建立隔离的闭卷验证集。
"""

import os
import random
import shutil

def main():
    base_dir = r"d:\vision_algo_workspace\vision-bootcamp\datasets\safety_helmet"
    train_img_dir = os.path.join(base_dir, "images", "train")
    train_lbl_dir = os.path.join(base_dir, "labels", "train")
    val_img_dir = os.path.join(base_dir, "images", "val")
    val_lbl_dir = os.path.join(base_dir, "labels", "val")

    if not os.path.exists(train_img_dir) or not os.path.exists(train_lbl_dir):
        print(f"❌ 找不到训练集路径，请先运行 auto_import.py 导入数据。")
        return

    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    all_imgs = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_count = len(all_imgs)
    
    if total_count == 0:
        print("❌ 训练集为空！")
        return

    val_ratio = 0.20
    val_count = int(total_count * val_ratio)
    print(f"📊 train 目录下共有 {total_count} 张图片，准备随机分出 {val_count} 张 (20%) 到验证集...")

    random.seed(42)  # 固定种子
    val_imgs = random.sample(all_imgs, val_count)

    moved_count = 0
    for img_name in val_imgs:
        src_img = os.path.join(train_img_dir, img_name)
        dst_img = os.path.join(val_img_dir, img_name)

        base_name = os.path.splitext(img_name)[0]
        lbl_name = base_name + ".txt"
        src_lbl = os.path.join(train_lbl_dir, lbl_name)
        dst_lbl = os.path.join(val_lbl_dir, lbl_name)

        if os.path.exists(src_img) and os.path.exists(src_lbl):
            shutil.move(src_img, dst_img)
            shutil.move(src_lbl, dst_lbl)
            moved_count += 1

    print(f"🎉 验证集划分完成！成功移动了 {moved_count} 对文件。")

if __name__ == "__main__":
    main()
