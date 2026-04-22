from sahi.slicing import slice_coco
import os

# ================= 配置区域 =================
COCO_JSON_PATH = "D:/SAVE/RAW_DATA/RAW_DATA4_10-4/coco_format.json"  # 上一步生成的 COCO json
IMAGE_DIR = "D:/SAVE/RAW_DATA/RAW_DATA4_10-4/images"                 # 原图文件夹
OUTPUT_DIR = "D:/SAVE/RAW_DATA/RAW_DATA4_10-4/sliced_dataset"        # 输出切片结果的文件夹

SLICE_SIZE = 640                       # 切片大小
OVERLAP_RATIO = 0.2                    # 重叠率
# ============================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("开始对数据集进行切片...")
    
    # 调用 SAHI 的核心切片函数
    coco_dict, coco_path = slice_coco(
        coco_annotation_file_path=COCO_JSON_PATH,
        image_dir=IMAGE_DIR,
        output_coco_annotation_file_name="train_sliced",
        ignore_negative_samples=True, # 设置为 False 保留没有目标的背景图切片，有助于降低模型误报
        output_dir=OUTPUT_DIR,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
        min_area_ratio=0.1,            # 过滤掉切片后面积过小的残缺目标
        verbose=False
    )
    
    print(f"\n切片完成！")
    print(f"新的切片图片和 COCO JSON 标签已保存至: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
