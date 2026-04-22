import os
from ultralytics import YOLO

# ================= 配置区域 =================
# 路径设置
IMAGE_DIR = 'D:/SAVE/RAW_DATA/RAW_DATA4_10-4/images'       # 输入：待预测的图片文件夹
PRED_DIR = 'D:/SAVE/RAW_DATA/RAW_DATA4_10-4/preds'         # 输出：保存预测结果 txt 的文件夹

# 模型设置
# 如果你有自己训练好的模型，请将路径替换为你的权重文件，例如 'runs/detect/train/weights/best.pt'
# 如果只是想测试，这里默认使用官方的轻量级预训练模型 'yolov8n.pt'，它会自动下载
MODEL_PATH = '20260409yolo26s.pt'    
# ============================================

def main():
    # 1. 确保预测结果输出目录存在
    os.makedirs(PRED_DIR, exist_ok=True)

    # 2. 加载 YOLO 模型
    print(f"正在加载模型: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)

    # 3. 获取所有待预测图片
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    if not os.path.exists(IMAGE_DIR):
        print(f"错误: 找不到图片文件夹 {IMAGE_DIR}")
        return

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"在 {IMAGE_DIR} 中没有找到图片文件。")
        return

    print(f"找到 {len(image_files)} 张图片，开始预测...")

    # 4. 遍历图片进行推理并保存为 TXT
    for img_name in image_files:
        img_path = os.path.join(IMAGE_DIR, img_name)
        base_name = os.path.splitext(img_name)[0]
        
        # 进行推理 (conf=0.25 表示置信度阈值，你可以根据需要调整)
        results = model.predict(source=img_path, conf=0.25, verbose=False)
        
        # 准备写入 txt 文件
        txt_out_path = os.path.join(PRED_DIR, f"{base_name}.txt")
        
        with open(txt_out_path, 'w', encoding='utf-8') as f:
            for result in results:
                # 获取边界框数据
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                
                # 遍历这张图片上的每一个预测框
                for i in range(len(boxes)):
                    # 获取类别 ID
                    cls_id = int(boxes.cls[i].item())
                    
                    # 获取归一化的中心点 x, y 和宽、高 (xywhn)
                    # 格式: [x_center, y_center, width, height]
                    x_c, y_c, w, h = boxes.xywhn[i].tolist()
                    
                    # 写入 txt 文件，格式为: class_id x_center y_center width height
                    # 注意：这里的 cls_id 并不重要，因为你在上一步的合并代码中已经强制将预测类别覆写为 "1" 了
                    f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
        
        print(f"已预测: {img_name} -> 发现 {len(results[0].boxes)} 个目标")

    print(f"\n全部预测完成！预测标签已保存至: {PRED_DIR}")
    print("现在你可以运行上一步的 JSON 合并脚本了。")

if __name__ == '__main__':
    main()
