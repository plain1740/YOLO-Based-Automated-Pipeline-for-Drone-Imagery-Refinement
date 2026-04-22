import os
import json
import shutil
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

# ================= 配置路径 =================
IMAGE_DIR = "E:/rivel_sample1/TRAIN_DATA3/train/images"      # 原始图片路径
LABEL_DIR = "E:/rivel_sample1/TRAIN_DATA3/train/labels"      # 原始YOLO标签路径
CSV_PATH = "E:/rivel_sample1/TRAIN_DATA3/anomalous_bboxes.csv" # 刚才生成的异常框CSV文件
OUTPUT_DIR = "E:/rivel_sample1/TRAIN_DATA3/anomaly_review"   # 输出提取图片和JSON的目标文件夹

def extract_and_convert_to_labelme():
    # 1. 检查并创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("正在加载异常框数据...")
    try:
        df_anomalies = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"无法读取CSV文件: {e}")
        return

    # 获取所有包含异常框的唯一图片名称
    anomalous_images = df_anomalies['image_name'].unique()
    print(f"共发现 {len(anomalous_images)} 张图片包含异常标注框，开始提取和转换...")

    for img_name in tqdm(anomalous_images):
        img_path = os.path.join(IMAGE_DIR, img_name)
        label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")
        
        if not os.path.exists(img_path) or not os.path.exists(label_path):
            continue

        # 2. 读取当前图片的异常记录
        # 提取出当前图片对应的所有异常框的 w_norm 和 h_norm
        img_anomaly_records = df_anomalies[df_anomalies['image_name'] == img_name]
        
        # 3. 读取图片获取宽高
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # 4. 初始化 LabelMe 格式的 JSON 结构
        label_data = {
            "version": "4.0.0-beta.2", 
            "flags": {}, 
            "shapes": [],
            "imagePath": img_name, 
            "imageData": None, 
            "imageHeight": h, 
            "imageWidth": w
        }

        # 读取原始 YOLO 标签
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # 5. 遍历该图片的所有原始 YOLO 预测框
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            # YOLO 格式: class_id, x_center, y_center, width, height
            c, x_c, y_c, w_norm, h_norm = map(float, parts[:5])
            
            # 计算绝对坐标 [xmin, ymin, xmax, ymax]
            xmin = max(0.0, (x_c - w_norm / 2) * w)
            ymin = max(0.0, (y_c - h_norm / 2) * h)
            xmax = min(float(w), (x_c + w_norm / 2) * w)
            ymax = min(float(h), (y_c + h_norm / 2) * h)

            # --- 核心逻辑：匹配异常框 ---
            # 通过长宽比例去匹配 CSV 中的异常框，加入极小的容差 (1e-4) 防止浮点数精度丢失导致匹配失败
            is_anomalous = False
            for _, row in img_anomaly_records.iterrows():
                if abs(row['w_norm'] - w_norm) < 1e-4 and abs(row['h_norm'] - h_norm) < 1e-4:
                    is_anomalous = True
                    break
            
            # 无问题的保持为 "0"，有问题的改为 "1"
            final_label = "1" if is_anomalous else "0"

            # 填入 shapes 列表
            label_data['shapes'].append({
                "label": final_label,  
                "points": [[xmin, ymin], [xmax, ymax]],
                "group_id": None, 
                "description": "",
                "difficult": False,
                "shape_type": "rectangle", 
                "flags": {},
                "attributes": {}
            })

        # 6. 保存提取的图片和生成的 JSON 文件到新文件夹
        target_img_path = os.path.join(OUTPUT_DIR, img_name)
        target_json_path = os.path.join(OUTPUT_DIR, os.path.splitext(img_name)[0] + ".json")

        # 拷贝图片
        shutil.copy2(img_path, target_img_path)

        # 写入 JSON
        with open(target_json_path, 'w', encoding='utf-8') as f:
            json.dump(label_data, f, indent=2, ensure_ascii=False)

    print(f"\n操作完成！所有包含异常框的图片及对应的 LabelMe JSON 文件已输出至:\n{OUTPUT_DIR}")

# 运行代码
if __name__ == "__main__":
    extract_and_convert_to_labelme()
