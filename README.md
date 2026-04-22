# UAV-Based-River-Debris-Detection / 无人机河道漂浮物检测

**A specialized data flywheel and modeling pipeline for UAV-based river garbage detection.**

一个专为无人机河道垃圾检测设计的自动化数据飞轮与模型优化工作流。

本系统从 0 到 1 构建了自动化数据处理与预标注流水线，包含基于频域分析的高质量抽帧、全局哈希去重、基于统计学 EDA 的数据清洗，以及引入 SAHI 的高精度离线预标注闭环。

## 🚀 核心特性 (Key Features)

### 1. Data Flywheel Pipeline (数据飞轮流水线)

- **Quality-Aware Extraction / 基于图像质量的智能抽帧:** Employs DWT (Discrete Wavelet Transform) in the frequency domain to evaluate and filter out low-quality frames caused by drone shaking or focus loss. / 针对无人机镜头晃动与失焦，引入离散小波变换（DWT）进行频域分析，精准筛选高质量帧。
    
- **Global Deduplication / 全局相似度去重:** Utilizes Global Hashing algorithms to automatically remove highly similar redundant frames, optimizing data diversity and reducing labeling costs. / 结合全局哈希（Global Hashing）算法，有效去除高度相似的冗余帧，大幅减少无效标注工作量。
    
- **Automated Pre-labeling Loop / 自动化预标注闭环:** Implemented a daily loop: "Frame Extraction $\rightarrow$ Best Model Pre-labeling $\rightarrow$ Manual Review". / 实现“每日抽帧 - 模型预标注 - 人工复核”闭环，随着数据集积累，模型预标注能力持续进化。
    

### 2. Model & Optimization (模型选型与优化)

- **Model Scaling Strategy / 轻量级模型架构:** Selected **YOLO26s** to balance training efficiency and accuracy. It perfectly matches the RTX 4080 hardware limits and prevents overfitting on early-stage small datasets. / 选用 YOLO26s，完美权衡 RTX 4080 算力限制与工程快速迭代需求，在初期百余张数据规模下有效防止过拟合。
    
- **Hyperparameter Evolution / 超参数进化:** Integrates Genetic Algorithms for phased hyperparameter tuning, yielding a stable ~5% boost in baseline mAP@50. / 阶段性引入遗传算法（Genetic Algorithm）进行超参数进化，打下坚实基线。
    
- **SAHI-Assisted Inference / 切片辅助超推理:** Integrates SAHI (Slicing Aided Hyper Inference) in the offline inference stage, boosting the pre-labeling mAP@50 to **86.2%**, significantly reducing manual verification pressure. / 在离线推理（预标注）端引入 SAHI，将预标注指标一路飙升至 mAP@50 86.2%。
    

### 3. Ablation & EDA-Driven Cleaning (基于统计学 EDA 的消融与清洗)

Through statistical EDA (bounding box area distribution, aspect ratio scatter plots, etc.), we identified key bottlenecks and applied targeted solutions:

通过脚本绘制多维度统计学 EDA 发现数据痛点，并进行了针对性的消融实验：

- **Small Target Filtering / 小目标过滤:** Removed extreme small targets (< 8x8 pixels) to improve training convergence stability. / 剔除 YOLO 下采样机制难以处理的极端小目标，提升训练收敛稳定性。
    
- **Outlier Cleaning / 异常点清洗:** Applied Random Forest algorithms to automatically clean up discrete outliers caused by the heavy-tailed distribution of garbage types. / 针对河道垃圾种类“重尾分布”导致的离散异常点，使用随机森林（Random Forest）进行自动化清洗。
    
- **Background Injection / 背景图注入:** Introduced pure background images to significantly reduce the model's false positive rate. / 引入纯背景图片，显著降低模型在复杂水面和树枝干扰下的误检率。
    

## 📝 待办事项与未来规划 (TODO & Roadmap)

### Short-term TODOs (近期任务)

- [ ] **Data Annotation / 数据标注:** Manual verification of the SAHI pre-labeled data. / 对高精度预标注数据进行人工校核。
    
- [ ] **Dataset Assembly / 数据集整理:** Format conversion for standard YOLO training. / 转化为标准的 YOLO 训练数据集格式。
    
- [ ] **Retraining / 新一轮训练:** Fine-tune the YOLO26s model with the newly refined data. / 使用新提取的精炼数据进行模型微调。
    

### Future Roadmap (长效规划)

- [ ] **Dataset Expansion / 数据集量级扩充:** Leverage the high-accuracy pre-labeling advantage to scale the dataset by another order of magnitude. / 利用 mAP 86.2% 的预标注优势，将数据集规模进一步提升一个数量级。
    
- [ ] **Unsupervised Clustering / 无监督聚类分析:** Perform unsupervised clustering on deduplicated samples to discover latent sub-categories of river garbage, aiming to break the current performance bottleneck. / 尝试对去重后的分类样本进行无监督聚类分析，探索河道垃圾的潜在细分类别规律，突破性能瓶颈。
    

## 📸 效果展示 (Showcase)

![[yolo_dataset_analysis.png]]
![[train_plot.png]]
![[result0.png]]