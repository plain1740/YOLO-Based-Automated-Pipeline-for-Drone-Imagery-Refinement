import os

def create_project_structure(base_path="."):
    """
    自动创建河道垃圾检测项目数据清洗工作流目录
    """
    # 定义目录列表（包含嵌套层级）
    directories = [
        "data/0_raw_video",           # 原始视频
        "data/1_raw_image",           # 初步抽帧结果
        "data/2_deduplicated_image/kept",             # 保留的唯一影像
        "data/2_deduplicated_image/label",            # 预标注文件
        "data/2_deduplicated_image/pairs_first_last", # 关键对记录
        "data/3_audit_past",
        "data/4_raw_train",          # 历史回检结果
        "data/5_train_data",
        "data/past_image",            # 历史图像库
        "data/special_background"      # 特殊背景负样本
    ]

    print(f"开始在路径 '{os.path.abspath(base_path)}' 下初始化目录结构...\n")

    for folder in directories:
        # 拼接完整路径
        path = os.path.join(base_path, folder)
        
        # 检查是否存在，不存在则创建
        if not os.path.exists(path):
            os.makedirs(path)
            print(f" [✓] 已创建: {folder}")
        else:
            print(f" [!] 已存在: {folder}")

    print("\n所有文件夹初始化完成！")

if __name__ == "__main__":
    # 你可以修改这里的路径，默认为当前脚本所在目录
    create_project_structure("../")
