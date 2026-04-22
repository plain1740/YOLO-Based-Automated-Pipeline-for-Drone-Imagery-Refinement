import os
import json

# ================= 配置区域 =================
# SAHI 切片后生成的完整 json 文件路径
COCO_JSON_PATH = './SAHI_DATA4_10-4/images/train_sliced_coco.json' 

# 输出 YOLO txt 标签的文件夹路径 (最好和切片图片放在同级的 labels 文件夹下)
OUTPUT_LABEL_DIR = './SAHI_DATA4_10-4/labels' 
# ============================================

def coco_to_yolo():
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    print(f"正在读取 COCO 文件: {COCO_JSON_PATH} ...")
    with open(COCO_JSON_PATH, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 创建一个字典，方便通过 image_id 快速查找图片的宽和高
    images_info = {img['id']: img for img in coco_data['images']}

    # 统计转换数量
    count = 0

    # 遍历所有的标注框
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        cls_id = ann['category_id']
        
        # COCO 的 bbox 格式是 [x_min, y_min, width, height]
        bbox = ann['bbox']
        x_min, y_min, w, h = bbox
        
        # 获取对应图片的宽和高，用于归一化
        img_info = images_info[img_id]
        img_w = img_info['width']
        img_h = img_info['height']
        img_filename = img_info['file_name']

        # 转换为 YOLO 格式: x_center, y_center, width, height (均需归一化 0~1)
        x_center = (x_min + w / 2) / img_w
        y_center = (y_min + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        # 确保坐标在 0-1 之间（修复可能超出图片边界的极个别脏数据）
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w_norm = max(0.0, min(1.0, w_norm))
        h_norm = max(0.0, min(1.0, h_norm))

        # 准备写入对应的 txt 文件
        base_name = os.path.splitext(img_filename)[0]
        txt_out_path = os.path.join(OUTPUT_LABEL_DIR, f"{base_name}.txt")

        # 追加写入模式 ('a')，因为同一张图可能有多个目标
        with open(txt_out_path, 'a', encoding='utf-8') as txt_file:
            txt_file.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        count += 1

    print(f"转换完成！共生成了 {count} 个目标框的 YOLO 标签。")
    print(f"标签保存在: {OUTPUT_LABEL_DIR}")

if __name__ == '__main__':
    coco_to_yolo()
