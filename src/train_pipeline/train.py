import gc
import torch
from ultralytics import YOLO
def main():
    task=['TRAIN_DATA5_4_vaild_40']
    for x in task:
        model = YOLO("yolo26s.pt")
        model.train(
             task= 'detect',
             data='D:/SAVE/TRAIN_DATA/'+x+'/data.yaml',
             name=x,
             epochs=200,
             batch=16,
            weight_decay=2.0e-05,
             workers=4
     )
    # --- 关键步骤：内存清理 ---
        del model
        gc.collect() # 强制进行垃圾回收
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # 清空 PyTorch 未使用的显存缓存
if __name__=="__main__":
    main()
