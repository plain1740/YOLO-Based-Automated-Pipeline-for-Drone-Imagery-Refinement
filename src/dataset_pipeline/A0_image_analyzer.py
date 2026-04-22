import os
import glob
import cv2
import numpy as np
import pandas as pd  # 用于数据导出
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from tqdm import tqdm

# ================= 配置路径 =================
IMAGE_DIR = "D:/SAVE/RAW_DATA/RAW_DATA4_10-4_sahi/images"  # 替换为你的图像文件夹路径
LABEL_DIR = "D:/SAVE/RAW_DATA/RAW_DATA4_10-4_sahi/labels"  # 替换为你的标签文件夹路径
OUTPUT_DIR = "D:/SAVE/RAW_DATA/RAW_DATA4_10-4_sahi/"

# ================= 辅助函数 =================
def compute_iou(box1, box2):
    """计算两个边界框的 交并比 (IoU)，输入格式为 [xmin, ymin, xmax, ymax]"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # 如果没有交集，直接返回 0.0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 计算交集和各自的面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集并返回 IoU
    union_area = box1_area + box2_area - intersection_area
    if union_area == 0:
        return 0.0
        
    return intersection_area / union_area

# ================= 主分析程序 =================
def analyzer(IMAGE_DIR, LABEL_DIR, OUTPUT_DIR):

    # ================= 数据收集 =================
    area_ratios = []       # 标注框面积占比
    aspect_ratios_w = []   # 标注框绝对宽度
    aspect_ratios_h = []   # 标注框绝对高度
    target_densities = []  # 每张图的bbox数量
    intra_image_ious = []  # 同图内部交并比
    brightness_means = []  # 亮度均值 (灰度均值)
    contrast_vars = []     # 对比度 (灰度方差)
    
    box_features = []      # 用于存储所有 bbox 信息以便导出用于无监督聚类

    # 支持的图片格式
    img_paths = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        img_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

    print(f"找到 {len(img_paths)} 张图片，开始分析...")

    for img_path in tqdm(img_paths):
        img_name = os.path.basename(img_path)
        label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")
        
        if not os.path.exists(label_path):
            continue

        # 读取图像（转为灰度图用于亮度和对比度计算）
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
            
        H_img, W_img = img_gray.shape

        # 读取标签
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        bboxes = []
        num_bboxes = len(lines)
        target_densities.append(num_bboxes) # 统计当前图片的bbox数量

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            # YOLO 格式: class_id, x_center, y_center, width, height (归一化)
            c, x_c, y_c, w_norm, h_norm = map(float, parts[:5])
            
            # 1. 面积比例 Area Ratio
            area_ratio = w_norm * h_norm
            area_ratios.append(area_ratio)
            
            # 计算绝对像素坐标
            w_box, h_box = w_norm * W_img, h_norm * H_img
            aspect_ratios_w.append(w_box)
            aspect_ratios_h.append(h_box)
            
            xmin = int(max(0, (x_c - w_norm / 2) * W_img))
            ymin = int(max(0, (y_c - h_norm / 2) * H_img))
            xmax = int(min(W_img, (x_c + w_norm / 2) * W_img))
            ymax = int(min(H_img, (y_c + h_norm / 2) * H_img))
            
            bboxes.append([xmin, ymin, xmax, ymax])
            
            # 2. 亮度均值与对比度（方差）提取
            brightness = 0.0
            contrast = 0.0
            if ymax > ymin and xmax > xmin:
                crop = img_gray[ymin:ymax, xmin:xmax]
                brightness = np.mean(crop)
                contrast = np.var(crop)
                brightness_means.append(brightness)
                contrast_vars.append(contrast)
                
            # 保存特征到字典，用于后续聚类分析
            box_features.append({
                "image_name": img_name,
                "class_id": int(c),
                "w_norm": w_norm,
                "h_norm": h_norm,
                "area_ratio": area_ratio,
                "w_pixel": w_box,
                "h_pixel": h_box,
                "brightness": brightness,
                "contrast": contrast,
                "boxes_in_image": num_bboxes
            })
        
        # 3. 同图内部交并比 (Intra-image IoU)
        if len(bboxes) > 1:
            # 组合当前图片中所有的 bbox 对
            for box1, box2 in combinations(bboxes, 2):
                iou = compute_iou(box1, box2)
                if iou > 0: # 通常只关注有重叠的
                    intra_image_ious.append(iou)

    # ================= 导出特征数据供聚类使用 =================
    if box_features:
        df_features = pd.DataFrame(box_features)
        export_path = os.path.join(OUTPUT_DIR, "bbox_features_for_clustering.txt")
        # 导出为制表符分割的txt，或者直接存为csv
        df_features.to_csv(export_path, sep='\t', index=False)
        print(f"\n数据已成功导出至: {export_path}，可用于无监督聚类异常检测！")

    # ================= 可视化绘图 =================
    print("开始生成可视化分析图表...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 12))

    # 1. 标注框面积分布 (Area Ratio Distribution)
    ax1 = plt.subplot(2, 3, 1)
    sns.histplot(area_ratios, bins=50, kde=True, color='skyblue', ax=ax1, log_scale=True)
    if area_ratios:
        area_median = np.median(area_ratios)
        ax1.axvline(area_median, color='red', linestyle='--', linewidth=2, label=f'Median: {area_median:.4f}')
    ax1.set_title("Bbox Area Ratio Distribution (Log Scale)", fontsize=12)
    ax1.set_xlabel("Area Ratio (Log Scale)")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    # 2. 带有阈值线的长宽比散点图 (Aspect Ratio Scatter Plot)
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(aspect_ratios_w, aspect_ratios_h, alpha=0.3, s=10, color='coral')
    if aspect_ratios_w and aspect_ratios_h:
        max_val = max(max(aspect_ratios_w), max(aspect_ratios_h))
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, label='1:1')
        ax2.plot([0, max_val], [0, max_val*2], 'g--', alpha=0.7, label='1:2 (W:H)')
        ax2.plot([0, max_val], [0, max_val*0.5], 'b--', alpha=0.7, label='2:1 (W:H)')
        ax2.set_xlim(0, max_val); ax2.set_ylim(0, max_val)
    ax2.set_title("Bbox Aspect Ratio Scatter Plot", fontsize=12)
    ax2.set_xlabel("Box Width (pixels)")
    ax2.set_ylabel("Box Height (pixels)")
    ax2.legend()

    # 3. 同图内部交并比 (Intra-image IoU)
    ax3 = plt.subplot(2, 3, 3)
    if intra_image_ious:
        sns.histplot(intra_image_ious, bins=30, kde=True, color='purple', ax=ax3)
    ax3.set_title("Intra-image IoU Distribution (>0)", fontsize=12)
    ax3.set_xlabel("IoU")
    ax3.set_ylabel("Frequency")

    # 4. 目标密度 (Bounding Boxes per Image)
    ax4 = plt.subplot(2, 3, 4)
    if target_densities:
        sns.histplot(target_densities, bins=max(10, min(50, max(target_densities))), color='green', ax=ax4)
        density_median = np.median(target_densities)
        q1, q3 = np.percentile(target_densities, [25, 75])
        iqr = q3 - q1
        upper_limit = density_median + 3 * iqr 
        
        # 限制X轴使其聚焦在中位分布区
        ax4.set_xlim(0, max(10, min(max(target_densities), upper_limit))) 
        ax4.axvline(density_median, color='red', linestyle='--', linewidth=2, label=f'Median: {density_median:.1f}')
    ax4.set_title("Target Density (Zoomed to Main Distribution)", fontsize=12)
    ax4.set_xlabel("Number of Bboxes")
    ax4.set_ylabel("Image Count")
    ax4.legend()

    # 5. 图像表现: 亮度和对比度双变量核密度图 (2D KDE Plot)
    ax5 = plt.subplot(2, 3, 5)
    if brightness_means and contrast_vars:
        sns.kdeplot(x=brightness_means, y=contrast_vars, cmap="Reds", fill=True, bw_adjust=0.5, ax=ax5)
    ax5.set_title("Bbox Brightness vs Contrast (2D KDE)", fontsize=12)
    ax5.set_xlabel("Mean Brightness (Grayscale)")
    ax5.set_ylabel("Contrast (Variance)")

    # 导出和展示
    plt.tight_layout()
    plot_output = os.path.join(OUTPUT_DIR, "yolo_dataset_analysis.png")
    plt.savefig(plot_output, dpi=300)
    print(f"分析绘图完成！图表已保存为 {plot_output}")
    plt.show()

if __name__ == '__main__':
    # 运行分析程序
    analyzer(IMAGE_DIR, LABEL_DIR, OUTPUT_DIR)
