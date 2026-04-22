import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_yolo_10_metrics(base_dir, output_filename="merged_10grid_results.png"):
    """
    遍历指定文件夹，提取多个 YOLO 实验的 CSV 数据，并绘制经典的 10 宫格对比图。
    """
    # 1. 查找所有 CSV 文件
    csv_files = list(Path(base_dir).rglob("*.csv"))
    
    if not csv_files:
        print(f"在 {base_dir} 及其子文件夹中没有找到任何 CSV 文件！")
        return

    # 2. 读取数据
    data_dict = {}
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # 关键步骤：清理 YOLO 数据列名自带的空格
            df.columns = df.columns.str.strip() 
            
            # 使用“父文件夹名_文件名”作为标识，例如 exp1_results
            label = f"{file.parent.name}_{file.stem}" 
            data_dict[label] = df
            print(f"成功读取: {file}")
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")

    if not data_dict:
        return

    # 3. 定义经典的 10 宫格指标顺序 (2行5列)
    # 第一行: train/box_loss, train/cls_loss, train/dfl_loss, metrics/precision, metrics/recall
    # 第二行: val/box_loss, val/cls_loss, val/dfl_loss, metrics/mAP50, metrics/mAP50-95
    target_metrics = [
        'train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'metrics/precision(B)', 'metrics/recall(B)',
        'val/box_loss',   'val/cls_loss',   'val/dfl_loss',   'metrics/mAP50(B)',     'metrics/mAP50-95(B)'
    ]

    # 4. 设置绘图风格和画布 (2行5列，尺寸设置宽一些以容纳5列)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(24, 10))
    axes = axes.flatten() # 将 2x5 的矩阵展平为一维数组，方便遍历

    # 5. 遍历数据并绘制到对应的子图中
    for label, df in data_dict.items():
        if 'epoch' not in df.columns:
            print(f"警告: {label} 中缺少 'epoch' 列，已跳过。")
            continue
            
        x = df['epoch']
        
        for i, metric in enumerate(target_metrics):
            ax = axes[i]
            if metric in df.columns:
                # 绘制曲线
                ax.plot(x, df[metric], label=label, linewidth=2, alpha=0.8)
            else:
                # 兼容处理：如果找不到某个特定的列，打印提示并在图中显示空白
                print(f"提示: {label} 中未找到列 '{metric}'")

    # 6. 美化每一个子图
    for i, metric in enumerate(target_metrics):
        ax = axes[i]
        # 只保留 / 后面的名字作为标题，例如 'train/box_loss' -> 'box_loss'
        title_name = metric.split('/')[-1] if '/' in metric else metric
        ax.set_title(title_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    # 7. 提取统一的图例 (只取第一个子图的图例来代表全局)
    handles, labels = axes[0].get_legend_handles_labels()
    # 将图例放在整个大图的顶部居中位置
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
               ncol=len(labels), fontsize=12, frameon=True, shadow=True)

    # 自动调整子图间距，防止文字重叠
    plt.tight_layout()
    
    # 8. 保存图片
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n绘图完成！图片已保存为: {output_filename}")

# ================= 运行 =================
if __name__ == "__main__":
    # 请将这里的路径修改为你存放多个 CSV 文件夹的总目录
    TARGET_DIRECTORY = "./runs/compare" 
    
    plot_yolo_10_metrics(TARGET_DIRECTORY)
