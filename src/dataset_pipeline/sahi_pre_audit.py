import os
import sys
import json
import shutil
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def process_analysis_with_sahi(
    model_path, 
    input_dir, 
    output_dir, 
    conf=0.5,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
):
    """
    使用 SAHI (切片推理) + YOLO 模型对图片进行预测，并将预测结果生成 LabelMe 格式的 JSON 文件
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("="*50)
    print("🚀 开始加载 SAHI 模型...")
    # 1. 初始化 SAHI 模型对象
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf,
        device="cuda:0",  # 如果没有GPU，可改为 "cpu"
    )
    print("✅ 模型加载完成！")
    print("="*50)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        json_name = os.path.splitext(img_name)[0] + '.json'
        
        # 2. 获取图片实际宽高 (LabelMe格式必需)
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            print(f"⚠️ 无法读取图像 {img_path}，已跳过。")
            continue
        h, w = img_cv.shape[:2]

        print(f"正在处理: {img_name} ...")

        # 3. 使用 SAHI 进行切片推理
        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            postprocess_type="NMM",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=0.5
        )
        
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

        # 5. 遍历 SAHI 的预测框，填入 shapes 列表
        # result.object_prediction_list 包含了所有合并后的预测框
        for object_prediction in result.object_prediction_list:
            bbox = object_prediction.bbox
            score = object_prediction.score.value
            
            # 你原来的代码里固定写死了 "1"，这里保持原样；如果需要实际类别名可用 object_prediction.category.name
            label_name = "1" 

            label_data['shapes'].append({
                "kie_linking": [],
                "label": label_name,  
                "score": float(score),
                # SAHI 的 bbox 提供了 minx, miny, maxx, maxy
                "points": [
                    [float(bbox.minx), float(bbox.miny)], 
                    [float(bbox.maxx), float(bbox.maxy)]
                ],
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

    print(f"\n✅ 处理完成！所有 SAHI 预测结果已保存至: {output_dir}")


if __name__ == "__main__":
    # ====================================================================
    # 全局配置参数 (结合了两份代码的配置)
    # ====================================================================
    MODEL_WEIGHTS = "20260410_sahi_yolo26s.pt"
    INPUT_DIR = "D:/SAVE/raw_image"
    OUTPUT_DIR = "D:/SAVE/raw_image"
    CONFIDENCE = 0.65  # SAHI 推理通常建议较低的置信度阈值
    
    # SAHI 切片参数配置
    SLICE_HEIGHT = 640
    SLICE_WIDTH = 640
    OVERLAP_HEIGHT_RATIO = 0.2
    OVERLAP_WIDTH_RATIO = 0.2
    
    # 支持命令行参数覆盖 (可选)
    if len(sys.argv) > 1:
        MODEL_WEIGHTS = sys.argv[1]
        INPUT_DIR = sys.argv[2]
        OUTPUT_DIR = sys.argv[3]
        if len(sys.argv) > 4:
            CONFIDENCE = float(sys.argv[4])
        
    # --- 执行脚本 ---
    process_analysis_with_sahi(
        model_path=MODEL_WEIGHTS, 
        input_dir=INPUT_DIR, 
        output_dir=OUTPUT_DIR, 
        conf=CONFIDENCE,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO
    )
