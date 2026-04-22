import os
import sys  # 补充了原代码缺失的 sys
import json
import shutil
import cv2
import numpy as np # 导入numpy
from ultralytics import YOLO
from shapely.geometry import box

def calculate_iou(box1, box2):
    """计算IoU [x1, y1, x2, y2]"""
    poly1 = box(*box1)
    poly2 = box(*box2)
    if not poly1.intersects(poly2):
        return 0.0
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.area + poly2.area - inter_area
    return inter_area / union_area

def process_analysis(model_path, input_dir, output_dir,
                     conf=0.5, threshold=0.5, check_undetected=False):
    model = YOLO(model_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        json_name = os.path.splitext(img_name)[0] + '.json'
        json_path = os.path.join(input_dir, json_name)
        
        # 预测结果
        results = model.predict(img_path, conf=conf, verbose=False)[0]
        
        # 【修改点 1】：同时获取预测框坐标和置信度 (score)
        pred_boxes_xyxy = results.boxes.xyxy.cpu().numpy().tolist() 
        pred_scores = results.boxes.conf.cpu().numpy().tolist()
        
        # 将坐标和置信度打包成列表以便后续遍历
        predictions = list(zip(pred_boxes_xyxy, pred_scores))

        is_changed = False
        label_data = None

        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            # 【可选修改】：为了保持数据结构一致，可以给原有的 ground truth 数据补全这些新字段
            for shape in label_data['shapes']:
                if 'score' not in shape: shape['score'] = None
                if 'kie_linking' not in shape: shape['kie_linking'] = []
                if 'description' not in shape: shape['description'] = ""
                if 'difficult' not in shape: shape['difficult'] = False
                if 'attributes' not in shape: shape['attributes'] = {}
            
            gt_boxes_coords = []
            for shape in label_data['shapes']:
                p = shape['points']
                xs, ys = [pt[0] for pt in p], [pt[1] for pt in p]
                curr_gt_box = [min(xs), min(ys), max(xs), max(ys)]
                gt_boxes_coords.append(curr_gt_box)
                
                # 计算漏检
                max_iou = max([calculate_iou(curr_gt_box, pb) for pb in pred_boxes_xyxy]) if len(pred_boxes_xyxy) > 0 else 0
                if max_iou < threshold:
                    shape['label'] = "2"
                    is_changed = check_undetected

            # 遍历预测框，加入新增检测
            for p_box, p_score in predictions:
                max_iou = max([calculate_iou(p_box, gb) for gb in gt_boxes_coords]) if len(gt_boxes_coords) > 0 else 0
                if max_iou < threshold:
                    # 【修改点 2】：参照目标格式，加入 score, kie_linking 等字段
                    new_shape = {
                        "kie_linking": [],
                        "label": "1",
                        "score": float(p_score),  # 填入 YOLO 预测出的置信度
                        "points": [[float(p_box[0]), float(p_box[1])], [float(p_box[2]), float(p_box[3])]], # 保持LabelMe矩形点对格式
                        "group_id": None, 
                        "description": "",
                        "difficult": False,
                        "shape_type": "rectangle", 
                        "flags": {},
                        "attributes": {}
                    }
                    label_data['shapes'].append(new_shape)
                    is_changed = True

        else:
            if len(predictions) > 0:
                is_changed = True
                img_cv = cv2.imread(img_path)
                h, w = img_cv.shape[:2]
                label_data = {
                    "version": "4.0.0-beta.2", "flags": {}, "shapes": [],
                    "imagePath": img_name, "imageData": None, "imageHeight": h, "imageWidth": w
                }
                for p_box, p_score in predictions:
                    # 【修改点 3】：新建 JSON 时也按目标格式加入相应字段
                    label_data['shapes'].append({
                        "kie_linking": [],
                        "label": "1",
                        "score": float(p_score), # 填入预测得分
                        "points": [[float(p_box[0]), float(p_box[1])], [float(p_box[2]), float(p_box[3])]],
                        "group_id": None, 
                        "description": "",
                        "difficult": False,
                        "shape_type": "rectangle", 
                        "flags": {},
                        "attributes": {}
                    })

        if is_changed and label_data:
            save_json_path = os.path.join(output_dir, json_name)
            with open(save_json_path, 'w', encoding='utf-8') as f:
                json.dump(label_data, f, indent=2, ensure_ascii=False)
            shutil.copy(img_path, os.path.join(output_dir, img_name))

    print(f"✅ 处理完成！差异数据已提取至: {output_dir}")


if __name__ == "__main__":
    input_dir = r"data/past_image"   
    output_dir = r"data/3_audit_past"
    model_path = r"models/20260313yolo26s.pt"
    conf=0.5
    threshold=0.5
    check_undetected=False
    
    # 配置你的路径
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        input_dir = sys.argv[2]  # 此处原代码变量名为 input_path，修正为 input_dir
        output_dir = sys.argv[3] # 此处原代码变量名为 output_path，修正为 output_dir
        conf = float(sys.argv[4]) # 确保转为 float
        threshold = float(sys.argv[5]) # 确保转为 float
        check_undetected = sys.argv[6].lower() == 'true' # 确保转为 bool
        
    # --- 执行设置 ---
    process_analysis(model_path, input_dir, output_dir, conf, threshold, check_undetected)
