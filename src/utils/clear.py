from pathlib import Path
import shutil

    directories = [
        "data/0_raw_video",           # 原始视频
        "data/1_raw_image",           # 初步抽帧结果
        "data/2_deduplicated_image",
        "data/3_audit_past",          # 历史回检结果
    ]
    
for folder in directories:
    if folder.exists() and folder.is_dir():
        shutil.rmtree(folder)
    else:
        pass
