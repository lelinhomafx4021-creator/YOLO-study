# -*- coding: utf-8 -*-
"""
auto_import.py - 一键自动搬运数据集
自动扫描 Windows 下载目录 C:\\Users\\19119\\Downloads 下解压出来的 Roboflow 安全帽数据集，
并物理拷贝到本项目的 datasets/safety_helmet 目录下。
"""

import os
import shutil

def main():
    target_img_dir = r"d:\vision_algo_workspace\vision-bootcamp\datasets\safety_helmet\images\train"
    target_lbl_dir = r"d:\vision_algo_workspace\vision-bootcamp\datasets\safety_helmet\labels\train"
    downloads_dir = r"C:\Users\19119\Downloads"

    if not os.path.exists(downloads_dir):
        print(f"❌ 找不到系统下载目录: {downloads_dir}")
        return

    print("🔍 正在扫描下载目录...")
    src_img_dir = None
    src_lbl_dir = None

    for root, dirs, files in os.walk(downloads_dir):
        if root.replace(downloads_dir, '').count(os.sep) > 3:
            continue
        if root.endswith("train") and "images" in dirs and "labels" in dirs:
            src_img_dir = os.path.join(root, "images")
            src_lbl_dir = os.path.join(root, "labels")
            print(f"✅ 找到源数据集: {root}")
            break

    if not src_img_dir or not src_lbl_dir:
        print("❌ 未能在 Downloads 目录下找到解压好的 Roboflow 数据集文件夹。")
        return

    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_lbl_dir, exist_ok=True)

    # 复制图片
    img_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"📊 开始搬运图片共 {len(img_files)} 张...")
    for file in img_files:
        shutil.copy2(os.path.join(src_img_dir, file), os.path.join(target_img_dir, file))

    # 复制标签
    lbl_files = [f for f in os.listdir(src_lbl_dir) if f.lower().endswith('.txt')]
    print(f"📊 开始搬运标签共 {len(lbl_files)} 个...")
    for file in lbl_files:
        shutil.copy2(os.path.join(src_lbl_dir, file), os.path.join(target_lbl_dir, file))

    print("\n🎉 数据一键搬运成功！")

if __name__ == "__main__":
    main()
