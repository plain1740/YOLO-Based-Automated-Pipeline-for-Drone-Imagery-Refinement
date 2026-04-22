import os
import json
import cv2
from ultralytics import SAM

def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """将 YOLO 的相对坐标转换为绝对像素的左上、右下坐标 (用于 LabelMe 的 rectangle)"""
    x_min = (cx - w / 2) * img_w
    y_min = (cy - h / 2) * img_h
    x_max = (cx + w / 2) * img_w
    y_max = (cy + h / 2) * img_h
    return [
        max(0, x_min), 
        max(0, y_min), 
        min(img_w, x_max), 
        min(img_h, y_max)
    ]

def auto_label_sam_to_json(image_dir, label_dir, output_dir, model_path="models/sam3.pt"):
    """
    自动读取图库与原有框标注，使用 SAM 3 分割，并输出 LabelMe 格式的 JSON。
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载 SAM 3 模型
    print(f"正在加载模型 {model_path}...")
    model = SAM(model_path)

    # 遍历图片文件夹
    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(image_dir, img_name)
        txt_path = os.path.join(label_dir, base_name + ".txt")
        json_path = os.path.join(output_dir, base_name + ".json")

        # 检查是否有对应的标签文件
        if not os.path.exists(txt_path):
            continue

        # 读取图像获取真实宽高
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
        img_h, img_w = img.shape[:2]

        bboxes = []
        shapes = []

        # 1. 读取并解析原有的 YOLO 标注文件
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # 获取相对坐标
                    cx, cy, w, h = map(float, parts[1:5])
                    xyxy = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
                    bboxes.append(xyxy)

                    # 将原有标注框写入 JSON (LabelMe的矩形格式为两点：左上角和右下角)
                    shapes.append({
                        "label": "0",
                        "score": None,
                        "points": [
                            [xyxy[0], xyxy[1]],
                            [xyxy[2], xyxy[3]]
                        ],
                        "group_id": None,
                        "description": "",
                        "difficult": False,
                        "shape_type": "rectangle",
                        "flags": {},
                        "attributes": {},
                        "kie_linking": []
                    })

        if not bboxes:
            continue

        # 2. 批量将所有的框传入 SAM 3 进行推理
        # verbose=False 可关闭终端里每张图的打印，加快进度条清晰度
        results = model(img_path, bboxes=bboxes, verbose=False)
        result = results[0]

        # 3. 提取 SAM 3 生成的多边形 (Polygons)
        if result.masks is not None:
            # result.masks.xy 返回的是一个列表，里面包含了每个目标的轮廓坐标组 [N, 2]
            for poly in result.masks.xy:
                # 只有包含 3 个点以上才构成有效多边形
                if len(poly) >= 3:
                    shapes.append({
                        "label": "1",          # SAM 输出的标注标记为 1
                        "score": None,
                        "points": poly.tolist(), # 将 numpy 数组转为标准 python 列表嵌套
                        "group_id": None,
                        "description": "",
                        "difficult": False,
                        "shape_type": "polygon",
                        "flags": {},
                        "attributes": {},
                        "kie_linking": []
                    })

        # 4. 构建 LabelMe 兼容的 JSON 结构
        labelme_json = {
            "version": "4.0.0-beta.2",
            "flags": {},
            "shapes": shapes,
            "imagePath": img_name,
            "imageData": None, # 不保存 base64 图像数据以节省空间
            "imageHeight": img_h,
            "imageWidth": img_w,
            "description": ""
        }

        # 5. 保存 JSON 文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_json, f, indent=2, ensure_ascii=False)

        print(f"✅ 成功生成标注: {json_path} (包含 {len(bboxes)} 个框, {len(result.masks.xy) if result.masks else 0} 个掩码)")

if __name__ == "__main__":
    # 配置您的目录路径
    IMAGE_DIR = "D:/SAVE/TRAIN_DATA/TRAIN_DATA4_isolation&10-4/train/images"       # 存放图片的目录
    LABEL_DIR = "D:/SAVE/TRAIN_DATA/TRAIN_DATA4_isolation&10-4/train/labels"       # 存放 YOLO .txt 的目录
    OUTPUT_DIR = "D:/SAVE/TRAIN_DATA/TRAIN_DATA4_isolation&10-4/train/images" # 生成的 LabelMe .json 存放目录
    
    # 开始执行
    auto_label_sam_to_json(IMAGE_DIR, LABEL_DIR, OUTPUT_DIR)
