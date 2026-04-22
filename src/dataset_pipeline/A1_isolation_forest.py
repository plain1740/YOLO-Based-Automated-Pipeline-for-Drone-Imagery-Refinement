import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ================= 配置路径 =================
# 1. 你的特征文件路径
FEATURE_DATA_PATH = "D:/SAVE/RAW_DATA/RAW_DATA4/bbox_features_for_clustering.txt"

# 2. 原始 YOLO 数据集的根目录 (包含 train 和 val 文件夹，你需要修改这里)
DATASET_ROOT = "D:/SAVE/RAW_DATA/RAW_DATA4/" 

# 3. 输出的新纯净数据集根目录
OUTPUT_DATASET_ROOT = "D:/SAVE/RAW_DATA/RAW_DATA6_isolation"

def filter_and_export_dataset(feature_path, dataset_root, output_root):
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("正在加载特征数据...")
    try:
        df = pd.read_csv(feature_path, sep='\t')
    except Exception as e:
        print(f"读取文件失败，请检查路径。错误: {e}")
        return

    if df.empty:
        print("数据为空！")
        return

    # ================= 1. 特征选择与异常检测 =================
    feature_cols = ['w_norm', 'h_norm', 'area_ratio', 'brightness', 'contrast', 'boxes_in_image']
    X = df[feature_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("正在训练 Isolation Forest 模型...")
    iso_forest = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    
    df['anomaly_label'] = iso_forest.fit_predict(X_scaled)
    df['anomaly_score'] = iso_forest.decision_function(X_scaled)

    # ================= 2. 构建异常框“黑名单” =================
    # 使用 图片名 + 类别 + 四位精度的宽高 来唯一锁定一个框
    anomalies = df[df['anomaly_label'] == -1]
    anomalous_set = set()
    
    for _, row in anomalies.iterrows():
        img_name = row['image_name']
        cls_id = str(int(row['class_id']))
        # 四舍五入到4位小数以防浮点数精度误差导致匹配失败
        w_norm = round(float(row['w_norm']), 4)
        h_norm = round(float(row['h_norm']), 4)
        anomalous_set.add((img_name, cls_id, w_norm, h_norm))

    print(f"\n分析完成！共检测到 {len(df)} 个标注框，判定异常并加入黑名单 {len(anomalous_set)} 个。")

    # ================= 3. 清理标签并生成新数据集 =================
    print("\n开始同步清理原数据集并提取正常图片与标签...")
    base_dir = Path(dataset_root)
    splits = [0]
    total_images_processed = 0
    total_boxes_removed = 0
    total_images_copied = 0

    for split in splits:
        img_folder = base_dir/ 'images'
        lbl_folder = base_dir / 'labels'
        
        dest_img_folder = out_dir / 'images'
        dest_lbl_folder = out_dir / 'labels'
        
        if not img_folder.exists() or not lbl_folder.exists():
            continue
            
        for img_path in img_folder.iterdir():
            if img_path.suffix.lower() not in {'.jpg', '.png', '.jpeg', '.bmp'}:
                continue
                
            total_images_processed += 1
            label_path = lbl_folder / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue
                
            valid_lines = []
            
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = parts[0]
                        w = round(float(parts[3]), 4)
                        h = round(float(parts[4]), 4)
                        
                        # 检查这个框是否在我们的异常黑名单里
                        if (img_path.name, cls_id, w, h) not in anomalous_set:
                            valid_lines.append(line)
                        else:
                            total_boxes_removed += 1
            
            # 如果清理后该图片仍有正常的框，复制到新数据集
            if valid_lines:
                dest_img_folder.mkdir(parents=True, exist_ok=True)
                dest_lbl_folder.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(img_path, dest_img_folder / img_path.name)
                
                with open(dest_lbl_folder / label_path.name, 'w', encoding='utf-8') as f_out:
                    f_out.writelines(valid_lines)
                    
                total_images_copied += 1

    print("\n" + "="*40)
    print("数据集清理与导出完成！")
    print(f"扫描图片总数: {total_images_processed}")
    print(f"成功切除的异常框数: {total_boxes_removed}")
    print(f"保留并复制的正常图片数: {total_images_copied}")
    print(f"纯净数据集保存在: {out_dir}")
    print("="*40)

    # ================= 4. 保存异常报表与 PCA 可视化 =================
    print("\n正在保存分析报表和 PCA 可视化图...")
    anomalies.sort_values(by='anomaly_score').to_csv(
        out_dir / 'anomalous_bboxes_report.csv', index=False, encoding='utf-8-sig'
    )
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df[df['anomaly_label'] == 1], 
        x='PCA1', y='PCA2', 
        alpha=0.3, color='blue', label='Normal Bboxes', s=20
    )
    sns.scatterplot(
        data=df[df['anomaly_label'] == -1], 
        x='PCA1', y='PCA2', 
        alpha=0.8, color='red', label='Anomalous Bboxes', s=50, marker='X'
    )
    
    plt.title("Bbox Anomaly Detection via Isolation Forest (PCA Projection)")
    plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = out_dir / 'anomaly_visualization.png'
    plt.savefig(plot_path, dpi=300)
    print(f"图表已保存至: {plot_path}")
    plt.close()

if __name__ == "__main__":
    filter_and_export_dataset(FEATURE_DATA_PATH, DATASET_ROOT, OUTPUT_DATASET_ROOT)
