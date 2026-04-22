import os
import json
import cv2

# ================= 配置区域 =================
IMAGE_DIR = 'D:/SAVE/RAW_DATA/RAW_DATA4_10-4/images'          # 你的图片文件夹
LABEL_DIR = 'D:/SAVE/RAW_DATA/RAW_DATA4_10-4/labels'          # 你的 YOLO txt 标签文件夹 (上一步生成的0或1的标签)
OUTPUT_JSON = 'D:/SAVE/RAW_DATA/RAW_DATA4_10-4/coco_format.json' # 输出的 COCO 格式文件路径

# 定义你的类别。根据你之前的需求，你有 0 和 1 两个类别
# 注意：COCO 格式的 category_id 通常建议从 1 开始，但为了和你原先的类别对应，这里直接映射
CLASSES = {
    0: "trash", # 原先有的类别
}
# ============================================

def yolo_to_coco():
    # COCO JSON 的基础结构
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 构建 categories 列表
    for cls_id, cls_name in CLASSES.items():
        coco_data["categories"].append({
            "id": cls_id,
            "name": cls_name,
            "supercategory": "none"
        })

    # 获取所有图片
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_exts)]
    
    ann_id = 1 # annotation 的唯一 ID，从 1 开始递增

    print(f"开始转换，共找到 {len(image_files)} 张图片...")

    for img_id, img_name in enumerate(image_files):
        img_path = os.path.join(IMAGE_DIR, img_name)
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(LABEL_DIR, f"{base_name}.txt")

        # 读取图片获取宽高
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图片 {img_path}，跳过。")
            continue
        
        img_h, img_w = img.shape[:2]

        # 1. 写入 image 信息
        coco_data["images"].append({
            "id": img_id, # 图片的唯一 ID
            "file_name": img_name,
            "width": img_w,
            "height": img_h
        })

        # 2. 如果不存在对应的标签文件，说明是负样本（背景图），直接跳过标签解析
        if not os.path.exists(label_path):
            continue

        # 3. 写入 annotation 信息
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            # 解析 YOLO 格式: class_id, x_center, y_center, width, height
            cls_id = int(parts[0])
            x_c, y_c, w_norm, h_norm = map(float, parts[1:5])

            # 还原为真实像素
            w = w_norm * img_w
            h = h_norm * img_h
            
            # 计算左上角坐标 (x_min, y_min)
            x_min = (x_c * img_w) - (w / 2)
            y_min = (y_c * img_h) - (h / 2)

            # 计算面积
            area = w * h

            # 组装标注信息
            coco_data["annotations"].append({
                "id": ann_id,
                "image_id": img_id, # 对应上面图片的 ID
                "category_id": cls_id,
                "bbox": [x_min, y_min, w, h], # COCO 格式 bbox: [x_min, y_min, width, height]
                "area": area,
                "segmentation": [], # 目标检测不需要分割掩码
                "iscrowd": 0        # 0 表示单个目标
            })
            ann_id += 1

    # 保存为 JSON 文件
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)

    print(f"\n转换完成！COCO 文件已保存至: {OUTPUT_JSON}")
    print(f"共转换了 {len(coco_data['images'])} 张图片，{len(coco_data['annotations'])} 个目标框。")

if __name__ == '__main__':
    yolo_to_coco()
