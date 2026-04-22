import os
import random
import shutil
from pathlib import Path

# ==========================================
# 函数 1: 单一文件夹标准划分与自适应 YAML 生成
# ==========================================
def split_single_dataset(src_dir, dst_dir, train_ratio=0.8):
    """
    将包含 images/ 和 labels/ 的目录按比例划分为 train/ 和 val/ 结构，并生成 data.yaml
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    src_img_dir = src_path / 'images'
    src_lbl_dir = src_path / 'labels'
    
    if not src_img_dir.exists() or not src_lbl_dir.exists():
        print(f"错误: 源目录 {src_dir} 缺少 images 或 labels 文件夹。")
        return

    # 获取所有图片文件
    valid_exts = {'.jpg', '.png', '.jpeg', '.bmp'}
    images = [f for f in src_img_dir.iterdir() if f.suffix.lower() in valid_exts]
    
    # 随机打乱以保证划分的随机性
    random.seed(42)
    random.shuffle(images)
    
    # 计算划分界限
    split_idx = int(len(images) * train_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    
    print(f"开始划分: 总计 {len(images)} 张图片。训练集 {len(train_imgs)} 张, 验证集 {len(val_imgs)} 张。")
    
    # 创建目标目录结构
    for split in ['train', 'val']:
        (dst_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (dst_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    def copy_data(img_list, split_name):
        copied_count = 0
        for img_path in img_list:
            lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
            
            # 只有当图片和标签同时存在时才复制
            if lbl_path.exists():
                shutil.copy2(img_path, dst_path / split_name / 'images' / img_path.name)
                shutil.copy2(lbl_path, dst_path / split_name / 'labels' / lbl_path.name)
                copied_count += 1
        return copied_count

    train_copied = copy_data(train_imgs, 'train')
    val_copied = copy_data(val_imgs, 'val')
    
    print(f"实际复制成功: 训练集 {train_copied} 对, 验证集 {val_copied} 对。")

    # 自适应生成 data.yaml
    # resolve().as_posix() 可以获取绝对路径并强制转换为 YOLO 兼容的正斜杠 '/'
    abs_dst_path = dst_path.resolve().as_posix()
    
    yaml_content = f"""# 训练集和验证集的根目录路径 (绝对路径)
path: {abs_dst_path}

# 相对于 path 的子路径
train: train
val: val

# 类别定义
nc: 1
names:
  0: ['trash']
"""
    yaml_file = dst_path / 'data.yaml'
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"成功生成自适应 data.yaml，路径配置为: {abs_dst_path}")
def strict_intersection_multi_folders(src_dirs_list, dst_dir, feature_filename="bbox_features_for_clustering.txt", train_ratio=0.8):
    """
    计算多个文件夹的严格交集，提取所有文件夹共有的标注框，并划分为标准 YOLO 数据集
    """
    print("=== 第一阶段：计算所有文件夹特征的严格交集 ===")
    global_valid_set = None

    for src_dir in src_dirs_list:
        src_path = Path(src_dir)
        img_folder = src_path / 'images'
        lbl_folder = src_path / 'labels'
        feature_file = src_path / feature_filename
        
        if not img_folder.exists() or not lbl_folder.exists() or not feature_file.exists():
            print(f"[警告] {src_path.name}: 目录不完整或缺失特征文件，已跳过。")
            continue
            
        current_folder_set = set()
        
        with open(feature_file, 'r', encoding='utf-8') as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 10:
                    img_name = parts[0]
                    cls_id = parts[1]
                    
                    img_path = img_folder / img_name
                    lbl_path = lbl_folder / f"{Path(img_name).stem}.txt"
                    
                    # 【核心 1】验证当前文件夹的物理文件确实存在
                    if img_path.exists() and lbl_path.exists():
                        w = round(float(parts[2]), 4)
                        h = round(float(parts[3]), 4)
                        current_folder_set.add((img_name, cls_id, w, h))
                        
        print(f"[扫描] {src_path.name}: 提取到 {len(current_folder_set)} 个物理存在的有效框。")
        
        # 【核心 2】执行严格交集 (Intersection)
        if global_valid_set is None:
            # 第一个文件夹，直接赋值作为基准
            global_valid_set = current_folder_set
        else:
            # 后续文件夹，只保留大家共有的框
            global_valid_set = global_valid_set.intersection(current_folder_set)

    if not global_valid_set:
        print("\n[错误] 所有文件夹的交集为空！没有任何框同时存在于所有传入的文件夹中。")
        return
        
    print(f"\n=== 交集计算完成：共有 {len(global_valid_set)} 个框在所有文件夹中完美重合 ===")

    # 将平铺的交集数据按 img_name 重新分组，方便后续还原完整的 YOLO 标签
    # 格式: {img_name: set((cls_id, w, h), ...)}
    valid_image_boxes = {}
    for (img_name, cls_id, w, h) in global_valid_set:
        if img_name not in valid_image_boxes:
            valid_image_boxes[img_name] = set()
        valid_image_boxes[img_name].add((cls_id, w, h))

    print(f"这些完美重合的框分布在 {len(valid_image_boxes)} 张图片中。")

    print("\n=== 第二阶段：还原完整 YOLO 标签并进行划分 ===")
    
    # 既然是交集，说明这些图片在所有源文件夹都存在。
    # 我们直接从列表里的第一个源文件夹提取真实的图片和完整的标签数据。
    base_src_path = Path(src_dirs_list[0])
    base_img_folder = base_src_path / 'images'
    base_lbl_folder = base_src_path / 'labels'

    # 打乱划分
    img_names = list(valid_image_boxes.keys())
    random.seed(42)
    random.shuffle(img_names)

    split_idx = int(len(img_names) * train_ratio)
    train_names = img_names[:split_idx]
    val_names = img_names[split_idx:]

    dst_path = Path(dst_dir)
    for split in ['train', 'val']:
        (dst_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (dst_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    def write_intersected_dataset(names_list, split_name):
        for img_name in names_list:
            src_img = base_img_folder / img_name
            src_lbl = base_lbl_folder / f"{Path(img_name).stem}.txt"
            
            # 复制原始图片
            shutil.copy2(src_img, dst_path / split_name / 'images' / img_name)
            
            valid_lines = []
            # 【核心 3】读取原标签，只提取命中了交集白名单的完整行 (带 x, y 中心点)
            with open(src_lbl, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = parts[0]
                        w = round(float(parts[3]), 4)
                        h = round(float(parts[4]), 4)
                        
                        # 匹配成功，保留整行
                        if (cls_id, w, h) in valid_image_boxes[img_name]:
                            valid_lines.append(line)
            
            # 写入清洗后的交集标签
            with open(dst_path / split_name / 'labels' / src_lbl.name, 'w', encoding='utf-8') as f_out:
                f_out.writelines(valid_lines)

    write_intersected_dataset(train_names, 'train')
    write_intersected_dataset(val_names, 'val')

    # 生成绝对路径的 data.yaml
    abs_dst_path = dst_path.resolve().as_posix()
    yaml_content = f"""# 训练集和验证集的根目录路径 (绝对路径)
path: {abs_dst_path}

# 相对于 path 的子路径
train: train
val: val

# 类别定义
nc: 1
names:
  0: ['trash']
"""
    with open(dst_path / 'data.yaml', 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\n全部完成！已成功提取各文件夹的严格交集。")
    print(f"训练集分配: {len(train_names)} 张，验证集分配: {len(val_names)} 张。")
    print(f"配置文件已生成: {abs_dst_path}/data.yaml")


# ==========================================
# 运行示例
# ==========================================
if __name__ == "__main__":
    split_single_dataset(
         src_dir="D:/SAVE/RAW_DATA/RAW_DATA_40", 
         dst_dir="D:/SAVE/TRAIN_DATA/TRAIN_DATA_40", 
         train_ratio=0.8
     )
'''
    # --- 测试函数 1 ---
    split_single_dataset(
         src_dir="D:/SAVE/RAW_DATA/RAW_DATA5_10-4", 
         dst_dir="D:/SAVE/TRAIN_DATA/TRAIN_DATA5_10-4", 
         train_ratio=0.8
     )
    split_single_dataset(
         src_dir="D:/SAVE/RAW_DATA/RAW_DATA6_isolation", 
         dst_dir="D:/SAVE/TRAIN_DATA/TRAIN_DATA6_isolation", 
         train_ratio=0.8
     )

    # --- 测试函数 2 ---
    # 填入你多个源文件夹的路径列表
    source_folders = [
        "D:/SAVE/RAW_DATA/RAW_DATA6_isolation",
        "D:/SAVE/RAW_DATA/RAW_DATA5_10-4",
    ]
 
    # 最终输出的高质量纯净数据集路径
    final_output_dir = "D:/SAVE/TRAIN_DATA/TRAIN_DATA7_isolation&10-4"
    
    strict_intersection_multi_folders(
        src_dirs_list=source_folders,
        dst_dir=final_output_dir,
        feature_filename="bbox_features_for_clustering.txt",
        train_ratio=0.8
    )
'''
