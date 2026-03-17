import os
import sys
import json
import shutil
import cv2
from ultralytics import YOLO

def process_analysis(model_path, input_dir, output_dir, conf=0.5):
    """
    使用YOLO模型对图片进行预测，并将预测结果生成LabelMe格式的JSON文件
    """
    model = YOLO(model_path)
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        json_name = os.path.splitext(img_name)[0] + '.json'
        
        # 1. YOLO 预测
        results = model.predict(img_path, conf=conf, verbose=False)[0]
        
        # 2. 获取预测框坐标和置信度 (score)
        pred_boxes_xyxy = results.boxes.xyxy.cpu().numpy().tolist() 
        pred_scores = results.boxes.conf.cpu().numpy().tolist()
        
        # 将坐标和置信度打包
        predictions = list(zip(pred_boxes_xyxy, pred_scores))

        # 3. 读取图片获取宽高（LabelMe格式必需）
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            print(f"⚠️ 无法读取图像 {img_path}，已跳过。")
            continue
        h, w = img_cv.shape[:2]

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

        # 5. 遍历预测框，填入 shapes 列表
        for p_box, p_score in predictions:
            label_data['shapes'].append({
                "kie_linking": [],
                "label": "1",  # 注意：这里目前固定写死为 "1"，如果有多类别需求可替换为 results.names[int(cls)]
                "score": float(p_score), # 预测得分
                "points": [[float(p_box[0]), float(p_box[1])], [float(p_box[2]), float(p_box[3])]],
                "group_id": None, 
                "description": "",
                "difficult": False,
                "shape_type": "rectangle", 
                "flags": {},
                "attributes": {}
            })

        # 6. 保存 JSON 文件
        save_json_path = os.path.join(output_dir, json_name)
        with open(save_json_path, 'w', encoding='utf-8') as f:
            json.dump(label_data, f, indent=2, ensure_ascii=False)
        
        # 7. 复制图片到输出目录（如果输入和输出目录相同，则跳过复制避免报错）
        save_img_path = os.path.join(output_dir, img_name)
        if os.path.abspath(img_path) != os.path.abspath(save_img_path):
            shutil.copy(img_path, save_img_path)

    print(f"✅ 处理完成！所有预测结果已保存至: {output_dir}")

if __name__ == "__main__":
    # 默认路径配置
    input_dir = r"data/2_deduplicated_image/kept"   
    output_dir = r"data/2_deduplicated_image/kept"
    model_path = r"models/20260313yolo26s.pt"
    conf = 0.7
    
    # 命令行参数配置
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        input_dir = sys.argv[2]
        output_dir = sys.argv[3]
        if len(sys.argv) > 4:
            conf = float(sys.argv[4])
        
    # --- 执行脚本 ---
    process_analysis(model_path, input_dir, output_dir, conf)
