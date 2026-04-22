import os
import glob
from sahi.predict import predict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ====================================================================
# 第一部分：全局配置参数
# ====================================================================
# 模型与数据路径
MODEL_WEIGHTS = "20260410_sahi_yolo26s.pt"
VAL_IMAGES_DIR = "D:/SAVE/RAW_DATA/RAW_DATA4_10-4/images"
GROUND_TRUTH_JSON = "D:/SAVE/RAW_DATA/RAW_DATA4_10-4/coco_format.json"

# 输出路径设置
EXPORT_DIR = "sahi_eval_pipeline"  # 总输出根目录
RUN_NAME = "val_run"               # 本次运行的文件夹前缀

os.makedirs(EXPORT_DIR, exist_ok=True)

# 运行前置检查
if not os.path.exists(MODEL_WEIGHTS):
    raise FileNotFoundError(f"找不到模型文件: {MODEL_WEIGHTS}")
if not os.path.exists(GROUND_TRUTH_JSON):
    raise FileNotFoundError(f"找不到真实标签文件: {GROUND_TRUTH_JSON}")


# ====================================================================
# 第二部分：SAHI 批量切片推理
# ====================================================================
def run_sahi_inference():
    print("="*50)
    print("🚀 阶段 1: 开始 SAHI 切片推理...")
    print("="*50)
    
    predict(
            model_type="ultralytics",
            model_path=MODEL_WEIGHTS,
            model_device="cuda:0",      
            model_confidence_threshold=0.01, 
            source=VAL_IMAGES_DIR,
            dataset_json_path=GROUND_TRUTH_JSON,
        
            
            # 切片参数
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            
            # 后处理参数
            postprocess_type="NMM",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=0.5,
            
            
            project=EXPORT_DIR,
            name=RUN_NAME,
            return_dict=False
        )

# ====================================================================
# 第三部分：自动寻找结果文件并计算 mAP
# ====================================================================
def run_pycocotools_evaluation():
    print("="*50)
    print("📊 阶段 2: 开始 pycocotools mAP 评估...")
    print("="*50)
    
    # 1. 动态寻找刚刚生成的 result.json
    # (应对 SAHI 自动创建 val_run, val_run2, val_run3 的情况)
    search_pattern = os.path.join(EXPORT_DIR, f"{RUN_NAME}*", "result.json")
    json_files = glob.glob(search_pattern)
    
    if not json_files:
        raise FileNotFoundError("未能找到 SAHI 生成的 result.json，请检查推理阶段是否有报错。")
    
    # 取修改时间最新的那个 json
    latest_pred_json = max(json_files, key=os.path.getmtime)
    print(f"正在读取最新的预测结果: {latest_pred_json}\n")
    
    # 2. 初始化 COCO API
    print("--> 加载真实标签 (Ground Truth)...")
    cocoGt = COCO(GROUND_TRUTH_JSON)
    
    print("--> 加载预测结果 (Detection)...")
    try:
        cocoDt = cocoGt.loadRes(latest_pred_json)
    except Exception as e:
        print(f"❌ 加载预测文件失败，通常是因为 Image ID 对不上: {e}")
        return
        
    # 3. 运行评估
    print("\n" + "-"*40)
    print("评估标准: bbox")
    print("-" * 40)
    
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    print("\n✅ 所有流程执行完毕！")


# ====================================================================
# 启动主程序
# ====================================================================
if __name__ == "__main__":
    # 1. 执行推理
    run_sahi_inference()
    
    # 2. 执行评估
    run_pycocotools_evaluation()
    print(f"{metric:<15}: {value:.4f}")
print("="*55)
