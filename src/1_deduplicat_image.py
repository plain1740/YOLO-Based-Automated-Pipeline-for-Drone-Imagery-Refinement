import cv2
import shutil
import numpy as np
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def get_image_hash(img_path):
    """预处理：读取并计算图片的均值哈希向量"""
    try:
        img = cv2.imread(str(img_path))
        if img is None: return None
        gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (32, 32))
        avg = gray.mean()
        return (gray > avg).flatten()
    except Exception:
        return None

def process_hover_frames(input_folder, output_base, threshold=0.85):
    # 创建两个分类文件夹
    base_dir = Path(output_base)
    kept_dir = base_dir / "kept"                 # 存放所有未删除的图片
    pairs_dir = base_dir / "pairs_first_last"    # 存放存在删除情况的首尾两张
    
    kept_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    exts = ['.jpg', '.jpeg', '.png', '.bmp']
    all_images = sorted([p for p in Path(input_folder).glob('*') if p.suffix.lower() in exts])
    
    if not all_images:
        print("未找到图片文件。")
        return

    print(f"🚀 正在并行计算 {len(all_images)} 张图片的特征值...")
    with ThreadPoolExecutor() as executor:
        hashes = list(executor.map(get_image_hash, all_images))

    print("📊 计算完成，开始分类保存（保持原名）...")

    kept_count = 0
    pairs_count = 0
    ref_idx = 0          # 当前序列的参考帧（首张）索引
    last_dup_idx = None  # 当前序列最后一张重复帧（尾张）索引

    for i in range(1, len(all_images)):
        if hashes[ref_idx] is None or hashes[i] is None: continue
        
        # 计算相似度
        similarity = np.mean(hashes[ref_idx] == hashes[i])

        if similarity > threshold:
            # 相似度高 -> 还在重复区间，更新尾张的索引
            last_dup_idx = i
        else:
            # 相似度低 -> 场景发生变化，序列中断
            
            # 1. 无论如何，当前的参考帧（首张）都属于“未删除的图片”，存入 kept 文件夹
            shutil.copy(all_images[ref_idx], kept_dir / all_images[ref_idx].name)
            kept_count += 1
            
            # 2. 如果发生了重复删除（last_dup_idx 不为空），将首尾张存入 pairs 文件夹
            if last_dup_idx is not None:
                shutil.copy(all_images[ref_idx], pairs_dir / all_images[ref_idx].name)
                shutil.copy(all_images[last_dup_idx], pairs_dir / all_images[last_dup_idx].name)
                pairs_count += 2
            
            # 更新参考索引，重置尾张
            ref_idx = i
            last_dup_idx = None

    # 处理最后一段画面
    shutil.copy(all_images[ref_idx], kept_dir / all_images[ref_idx].name)
    kept_count += 1
    if last_dup_idx is not None:
        shutil.copy(all_images[ref_idx], pairs_dir / all_images[ref_idx].name)
        shutil.copy(all_images[last_dup_idx], pairs_dir / all_images[last_dup_idx].name)
        pairs_count += 2

    print(f"✨ 处理完毕！")
    print(f"📂 [kept] 文件夹：共保存未删除的图片 {kept_count} 张。")
    print(f"📂 [pairs_first_last] 文件夹：共提取首尾图片 {pairs_count} 张（{pairs_count//2} 对）。")

if __name__ == "__main__":
    input_path = r"data/1_raw_image"   
    output_path = r"data/2_deduplicated_image" 
    threshold=0.6
    if len(sys.argv) > 1:
        input_path = sys.argv[1] 
        output_path = sys.argv[2]
        threshold = float(sys.argv[3])
    process_hover_frames(input_path, output_path, threshold)

