import os
import random
import shutil
from pathlib import Path

def prepare_yolo_dataset(
    unsplit_base_dir,    # 未划分数据集的根目录 (包含 image 和 label)
    normal_bg_dir,       # 普通背景图目录
    special_bg_dir,      # 特殊背景图目录
    output_base_dir,     # 输出的根目录
    val_ratio=0.2,
    bg_ratio=0.1
):
    """
    读取指定格式的输入目录，按比例抽取背景图，并输出为 YOLO 标准格式。
    """
    # 1. 解析输入目录
    unsplit_images_dir = Path(unsplit_base_dir) / 'image'
    unsplit_labels_dir = Path(unsplit_base_dir) / 'label'

    if not unsplit_images_dir.exists() or not unsplit_labels_dir.exists():
        print(f"错误: 找不到输入目录！请确保 '{unsplit_base_dir}' 下包含 'image' 和 'label' 文件夹。")
        return

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    def get_image_files(directory):
        return [f for f in Path(directory).iterdir() if f.suffix.lower() in valid_exts]

    # 2. 创建标准的输出目录结构 (train/val 下分 images/labels)
    base_out = Path(output_base_dir)
    for split in ['train', 'val']:
        for folder in ['images', 'labels']:
            (base_out / split / folder).mkdir(parents=True, exist_ok=True)

    # 3. 获取未划分训练集 (带目标)
    unsplit_images = get_image_files(unsplit_images_dir)
    num_unsplit = len(unsplit_images)
    
    if num_unsplit == 0:
        print("未在输入的 image 目录中找到图像！")
        return

    # 4. 计算需要加入的背景图总数
    num_bg_needed = int(num_unsplit * bg_ratio)

    # 5. 获取并分配背景图抽取名额
    normal_bgs = get_image_files(normal_bg_dir)
    special_bgs = get_image_files(special_bg_dir)
    total_avail_bg = len(normal_bgs) + len(special_bgs)

    if total_avail_bg < num_bg_needed:
        print(f"警告: 背景图库存 ({total_avail_bg}) 小于所需 ({num_bg_needed})。将抽取全部。")
        num_bg_needed = total_avail_bg

    if total_avail_bg > 0:
        special_weight = 0.7
        num_special = min(int(num_bg_needed * special_weight),len(special_bgs))
        num_normal = num_bg_needed - num_special
    else:
        print("警告: 未找到任何背景图像！将只划分原数据集。")
        num_normal, num_special = 0, 0

    # 6. 随机抽取背景图
    sampled_normal_bgs = random.sample(normal_bgs, num_normal)
    sampled_special_bgs = random.sample(special_bgs, num_special)

    # 7. 打标签并合并 (使用元组记录：(图片路径, 是否为背景图))
    data_with_flags = [(img, False) for img in unsplit_images] + \
                      [(bg, True) for bg in sampled_normal_bgs + sampled_special_bgs]

    random.seed(42) # 保证每次划分结果一致
    random.shuffle(data_with_flags)

    # 8. 划分训练集和验证集
    num_val = int(len(data_with_flags) * val_ratio)
    val_data = data_with_flags[:num_val]
    train_data = data_with_flags[num_val:]

    # 9. 核心处理函数：复制图片和处理标签
    def process_and_copy(data_list, split_name):
        img_dest = base_out / split_name / 'images'
        lbl_dest = base_out / split_name / 'labels'
        
        for img_path, is_background in data_list:
            # 复制图片
            shutil.copy(img_path, img_dest / img_path.name)
            
            # 处理标签
            label_name = img_path.stem + '.txt'
            lbl_out_path = lbl_dest / label_name
            
            if is_background:
                # 背景图生成空的 txt 文件
                open(lbl_out_path, 'w').close()
            else:
                # 带目标的图，从原 label 目录复制 txt 文件
                src_label = unsplit_labels_dir / label_name
                if src_label.exists():
                    shutil.copy(src_label, lbl_out_path)
                else:
                    print(f"警告: 找不到图片 {img_path.name} 对应的标签文件 {src_label.name}")

    # 执行复制
    print("正在生成 train 目录文件...")
    process_and_copy(train_data, 'train')
    
    print("正在生成 val 目录文件...")
    process_and_copy(val_data, 'val')

    print("数据集处理与划分完成！")

# ================= 运行示例 =================
unsplit_base_dir = "data/4_raw_train/"
normal_bg_dir = "data/past_background"
special_bg_dir = "data/special_background"
output_base_dir = "data/5_train_data"      

prepare_yolo_dataset(unsplit_base_dir, normal_bg_dir, special_bg_dir, output_base_dir)

