import shutil
from pathlib import Path

def sync_filter_dataset(txt_path, dataset_root, output_dir, target_size=640, min_box=8):
    input_file = Path(txt_path)
    base_dir = Path(dataset_root)
    out_dir = Path(output_dir)
    
    # 建立输出根目录
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt_file = out_dir / "filtered_bbox_features.txt"
    
    # 计算归一化阈值 (8 / 640 = 0.0125)
    norm_threshold = min_box / target_size
    
    valid_image_names = set()
    kept_features_count = 0
    
    # ==========================================
    # 步骤 1: 过滤 bbox 特征文件，并记录合格的图片名称
    # ==========================================
    print(f"--- 1. 正在过滤特征文件: {input_file.name} ---")
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(out_txt_file, 'w', encoding='utf-8') as f_out:
        
        header = f_in.readline()
        f_out.write(header)
        
        for line in f_in:
            parts = line.strip().split('\t')
            if len(parts) < 10:
                continue
                
            image_name = parts[0]
            w_norm = float(parts[2])
            h_norm = float(parts[3])
            
            # 如果符合 8x8 标准，则写入新文件并记录图片名
            if w_norm >= norm_threshold and h_norm >= norm_threshold:
                f_out.write(line)
                valid_image_names.add(image_name)
                kept_features_count += 1
                
    print(f"特征过滤完成！保留了 {kept_features_count} 个合格框特征。")
    print(f"共有 {len(valid_image_names)} 张图片包含合格目标。\n")

    # ==========================================
    # 步骤 2: 遍历 YOLO 数据集，清理对应的 labels 并复制图片
    # ==========================================
    print("--- 2. 正在同步清理 YOLO 标签并复制数据 ---")
    copied_images_count = 0
    
    img_folder = base_dir / 'images'
    lbl_folder = base_dir / 'labels'
        
    # 延迟创建目标目录
    dest_img_folder = out_dir / 'images'
    dest_lbl_folder = out_dir /  'labels'
        
      
    for img_path in img_folder.iterdir():
            # 只有在 valid_image_names 集合里的图片，我们才处理
            if img_path.name in valid_image_names:
                label_path = lbl_folder / f"{img_path.stem}.txt"
                
                if not label_path.exists():
                    continue
                    
                valid_label_lines = []
                
                # 读取并清洗 YOLO 标签文件
                with open(label_path, 'r', encoding='utf-8') as f_lbl:
                    for line in f_lbl:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            w = float(parts[3])
                            h = float(parts[4])
                            # 使用同样的阈值剔除过小的框
                            if w >= norm_threshold and h >= norm_threshold:
                                valid_label_lines.append(line)
                
                # 如果清洗后该图片仍有合格框，则复制图片并保存新标签
                if valid_label_lines:
                    dest_img_folder.mkdir(parents=True, exist_ok=True)
                    dest_lbl_folder.mkdir(parents=True, exist_ok=True)
                    
                    # 复制原图片
                    shutil.copy2(img_path, dest_img_folder / img_path.name)
                    
                    # 写入过滤后的新标签
                    with open(dest_lbl_folder / label_path.name, 'w', encoding='utf-8') as f_out_lbl:
                        f_out_lbl.writelines(valid_label_lines)
                        
                    copied_images_count += 1

    print(f"--- 数据同步完成 ---")
    print(f"成功复制图片及对应清洗后的标签: {copied_images_count} 对")
    print(f"干净的数据集已保存在: {out_dir}")

if __name__ == "__main__":
    # 你的特征 txt 文件路径
    TXT_FILE = "D:/SAVE/RAW_DATA/RAW_DATA4/bbox_features_for_clustering.txt" 
    
    # 原始 YOLO 数据集的根目录 (包含 train 和 val)
    DATASET_ROOT = "D:/SAVE/RAW_DATA/RAW_DATA4"
    
    # 输出的全新纯净数据集目录
    OUTPUT_ROOT = "D:/SAVE/RAW_DATA/RAW_DATA5_10-4/"
    
    sync_filter_dataset(
        txt_path=TXT_FILE, 
        dataset_root=DATASET_ROOT, 
        output_dir=OUTPUT_ROOT, 
        target_size=640, 
        min_box=8
    )

