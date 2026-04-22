import os
import json
import shutil
import cv2

# ================= 配置区域 =================
# 输入文件夹路径 (请根据实际情况修改)
IMAGE_DIR = 'D:/SAVE/RAW_DATA/RAW_DATA4_10-4/images'                # 原始图片文件夹
LABEL_GT_DIR = 'D:/SAVE/RAW_DATA/RAW_DATA4_10-4/labels'             # 真实标签文件夹 (归为类别 "0")
LABEL_PRED_DIR = 'D:/SAVE/RAW_DATA/RAW_DATA4_10-4/preds'            # 模型预测标签文件夹 (归为类别 "1", 如果没有可设为 None 或保持空文件夹)

# 输出文件夹路径
OUTPUT_DIR = 'D:/SAVE/AUDIT_DATA/AUDIT_DATA4_10-4/'               # 输出根目录
OUT_IMAGE_DIR = OUTPUT_DIR
OUT_JSON_DIR = OUTPUT_DIR
# ============================================

def parse_yolo_txt(txt_path, img_w, img_h, target_label_name):
    """
    解析 YOLO 格式的 txt 文件并转换为目标 JSON 的 shapes 列表。
    支持 Bbox (5个数字) 和 Polygon (多于5个数字)
    """
    shapes = []
    if not os.path.exists(txt_path):
        return shapes
        
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
            
        # YOLO 格式: [class_id, x, y, w, h] 或 [class_id, x1, y1, x2, y2, ...]
        if len(parts) == 5: 
            # 这是一个 Bounding Box (矩形)
            _, cx, cy, w, h = map(float, parts)
            # 还原真实像素坐标
            xmin = (cx - w / 2) * img_w
            xmax = (cx + w / 2) * img_w
            ymin = (cy - h / 2) * img_h
            ymax = (cy + h / 2) * img_h
            
            # 按照你提供的格式，矩形由4个点组成
            points = [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax]
            ]
            shape_type = "rectangle"
            
        elif len(parts) > 5:
            # 这是一个 Polygon (多边形)
            coords = list(map(float, parts[1:]))
            points = []
            for i in range(0, len(coords), 2):
                points.append([coords[i] * img_w, coords[i+1] * img_h])
            shape_type = "polygon"
        else:
            continue
            
        # 构建单个 shape 字典
        shape = {
            "label": str(target_label_name),
            "score": None, # Python的 None 转换为 JSON 时会变成 null
            "points": points,
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": shape_type,
            "flags": {},
            "attributes": {},
            "kie_linking": []
        }
        shapes.append(shape)
        
    return shapes

def main():
    # 创建输出目录
    os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUT_JSON_DIR, exist_ok=True)
    
    # 获取所有图片
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]
    
    for img_name in image_files:
        img_path = os.path.join(IMAGE_DIR, img_name)
        base_name = os.path.splitext(img_name)[0]
        
        # 读取图片以获取宽高
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图片 {img_path}，已跳过。")
            continue
        img_h, img_w = img.shape[:2]
        
        # 定义对应的标签文件路径
        gt_txt_path = os.path.join(LABEL_GT_DIR, f"{base_name}.txt")
        pred_txt_path = os.path.join(LABEL_PRED_DIR, f"{base_name}.txt") if LABEL_PRED_DIR else ""
        
        # 解析真实标签 (赋予类别 "0")
        shapes_gt = parse_yolo_txt(gt_txt_path, img_w, img_h, target_label_name="0")
        
        # 解析预测标签 (赋予类别 "1")
        shapes_pred = parse_yolo_txt(pred_txt_path, img_w, img_h, target_label_name="1") if LABEL_PRED_DIR else []
        
        # 合并所有的 shapes
        all_shapes = shapes_gt + shapes_pred
        
        # 构建最终的 JSON 结构
        json_data = {
            "version": "4.0.0-beta.2",
            "flags": {},
            "shapes": all_shapes,
            "imagePath": f"../images/{img_name}", # 根据你使用的工具可能需要调整该相对路径
            "imageData": None,
            "imageHeight": img_h,
            "imageWidth": img_w
        }
        
        # 复制图片到输出目录
        out_img_path = os.path.join(OUT_IMAGE_DIR, img_name)
        shutil.copy(img_path, out_img_path)
        
        # 写入 JSON 文件
        out_json_path = os.path.join(OUT_JSON_DIR, f"{base_name}.json")
        with open(out_json_path, 'w', encoding='utf-8') as f:
            # indent=2 保证格式化输出好看
            json.dump(json_data, f, indent=2, ensure_ascii=False)
            
        print(f"成功处理: {img_name} -> 包含 {len(shapes_gt)} 个原标签(0), {len(shapes_pred)} 个预测标签(1)")

if __name__ == '__main__':
    main()
