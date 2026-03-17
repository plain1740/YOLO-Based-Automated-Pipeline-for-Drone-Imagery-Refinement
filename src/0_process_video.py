import cv2
import os
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import sys

def extract_frames(video_path, save_dir, interval):
    """
    独立抽帧函数，设计为支持多进程调用
    """
    try:
        # 使用支持中文路径的读取方式 (如果使用 OpenCV 4.x 以上，str(video_path) 通常已原生支持中文，但保险起见保持原样)
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"  ❌ 无法打开 (跳过): {video_path.name}")
            return video_path.name, False, 0

        video_stem = video_path.stem
        frame_count = 0
        saved_count = 0

        while True:
            # 优化点 1: 只抓取不解码
            # grab() 比 read() 快非常多，因为它只将帧取到内存，但不做耗时的解码操作
            ret = cap.grab()
            if not ret:
                break
            
            # 只有当到达目标间隔时，才进行真正的解码 (retrieve)
            if frame_count % interval == 0:
                ret, frame = cap.retrieve()
                if ret:
                    # 文件名：视频原名_序号.jpg
                    img_name = f"{video_stem}_{saved_count:05d}.jpg"
                    img_save_path = save_dir / img_name
                    
                    # 使用 imencode 解决中文路径保存问题
                    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if success:
                        with open(img_save_path, "wb") as f:
                            f.write(buffer)
                        saved_count += 1
            
            frame_count += 1

        cap.release()
        return video_path.name, True, saved_count
        
    except Exception as e:
        print(f"  ❌ 处理 {video_path.name} 时发生错误: {e}")
        return video_path.name, False, 0


def process_all_videos(input_root, output_root, interval=60):
    input_path = Path(input_root).resolve()
    output_path = Path(output_root).resolve()
    
    # 用于收集所有需要处理的视频任务
    tasks = []

    # 1. 遍历并收集任务
    for item in input_path.iterdir():
        if item.is_dir():
            target_category_dir = output_path 
            video_files = list(item.rglob("*.mp4"))
            
            if not video_files:
                continue
                
            print(f"📁 发现分类目录: {item.name} | 找到 {len(video_files)} 个视频")
            

            # 将任务加入列表：(视频路径, 保存目录, 间隔)
            for video_file in video_files:
                tasks.append((video_file, output_path, interval))

    if not tasks:
        print("没有找到任何需要处理的视频！")
        return

    # 2. 优化点 2: 开启多进程池并发处理
    # 默认使用系统所有可用的 CPU 核心数
    max_workers = max(1, multiprocessing.cpu_count() - 1) # 留一个核心给系统，避免电脑卡死
    print(f"\n🚀 启动多进程处理，启用 {max_workers} 个工作进程...\n")

    # 使用 ProcessPoolExecutor 并行处理视频
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(extract_frames, task[0], task[1], task[2]) for task in tasks]
        
        # 监控进度
        for future in as_completed(futures):
            video_name, success, count = future.result()
            if success:
                print(f"  ✅ 已完成: {video_name} -> 提取 {count} 张")


if __name__ == "__main__":
    input_path = r"data/0_raw_video"   # 请替换为你的实际路径
    output_path = r"data/1_raw_image" # 请替换为你的实际路径
    interval = 60
    # 配置你的路径
    if len(sys.argv) > 1:
        input_path= sys.argv[1] 
        output_path= sys.argv[2]
        interval= sys.argv[3] 
    # 配置你的路径
    process_all_videos(input_path, output_path, interval)
