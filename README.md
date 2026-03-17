# YOLO-Based Automated Pipeline for Drone Imagery Refinement / 基于YOLO的无人机影像自动化处理

A specialized pipeline for river garbage detection, featuring automated frame extraction, similarity-based pruning, and model-assisted pre-labeling.

一个专为河道垃圾检测设计的自动化处理工作流，包含视频抽帧、相似度去重、旧模型漏检回溯及新数据预标注等核心环节。

---

## 🚀 核心特性 (Key Features)

- **Smart Frame Extraction / 智能抽帧**: High-speed extraction of image frames from raw drone video footage. / 从原始无人机视频中高效提取图像素材。
    
- **Duplicate Removal / 相似度过滤**: Automatic deletion of redundant images using perceptual hashing to reduce labeling costs. / 通过感知哈希等算法自动删除高相似度图片，减少冗余标注工作量。
    
- **Iterative Refinement / 迭代优化**:
    
    - **Back-testing**: Use the best model from the previous day to detect omissions in historical data. / **漏检回溯**：利用前一天的最优模型对历史素材进行回检，捕捉错漏。
        
    - **Pre-labeling**: Auto-generate annotations for new datasets to accelerate the labeling process. / **预标注**：对待标注素材进行模型预推断，提高标注效率。
        

---

## 📁 项目结构 (Project Structure)


```text
.
├── data/
│   ├── 0_raw_video/          # Source video files (.mp4, .avi) / 原始视频
│   ├── 1_raw_image/          # Initially extracted frames / 初步抽帧结果
│   ├── 2_deduplicated_image/ # Cleaned data / 去重后的数据
│   │   ├── kept/             # High-quality unique images / 保留的唯一影像
│   │   ├── label/            # Auto-generated labels / 预标注文件
│   │   └── pairs_first_last/ # Metadata for reference / 关键对记录
│   ├── 3_audit_past/         # Audit results from previous models / 历史回检结果
│   ├── past_image/           # Historical image pool / 历史图像库
│   └── special_background/   # Negative samples for background / 特殊背景负样本
├── models/                   # Storage for .pt or .onnx weights / 存放各版本权重文件
├── src/                      # Core algorithms and scripts / 核心处理代码
├── run.bat                   # One-click Windows startup / Windows一键启动脚本
└── requirements.txt          # Python dependencies / 项目依赖清单
````
---

## 🛠️ 安装指南 (Installation)

Bash

```
# Clone the repository / 克隆仓库
git clone https://github.com/plain1740/YOLO-Based-Automated-Pipeline-for-Drone-Imagery-Refinement.git
cd Drone Imagery Refinement

# Install dependencies / 安装依赖
pip install -r requirements.txt
```

---

## ⚡ 快速开始 (Quick Start)

1. **Prepare Data**: Place your `.mp4` or `.avi` files into `data/0_raw_video/`.
    
    **准备数据**：将视频文件放入 `data/0_raw_video/` 文件夹。
    
2. **Run Pipeline**: Double-click `run.bat` in the root directory to start the automated process.
    
    **运行流程**：双击根目录下的 `run.bat` 开启全自动处理流水线。
    
3. **Check Results**: Refined images and pre-labels will be available in `data/3_pre_labeled/`.
    
    **查看结果**：优化后的图片与自动标注文件将生成在 `data/3_pre_labeled/` 中。
    

---

## 📝 待办事项 (TODO)

- [ ] **Data Annotation / 数据标注**: Manual verification of pre-labeled data. / 对预标注数据进行人工校核。
    
- [ ] **Dataset Assembly / 数据集整理**: Format conversion for YOLO training. / 转化为标准的 YOLO 训练数据集格式。
    
- [ ] **Retraining / 新一轮训练**: Fine-tune the model with the newly refined data. / 使用新提取的精炼数据进行模型微调。
### 5. 效果展示 (Showcase)

![[Pasted image 20260317085454.png]]
![[Pasted image 20260317085505.png]]
![[Pasted image 20260317085518.png]]
![[Pasted image 20260317085546.png]]
![[Pasted image 20260317085645.png]]