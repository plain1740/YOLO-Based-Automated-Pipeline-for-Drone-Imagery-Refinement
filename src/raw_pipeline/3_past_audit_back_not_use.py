import os
import json
import shutil

def refine_labels_logic(target_dir, folder_1, folder_2):
    """
    target_dir: 分析结果文件夹 (包含图片和JSON)
    folder_1: 1号文件夹 (存放包含类别3且转换后的结果)
    folder_2: 2号文件夹 (存放清理后为空的结果)
    """
    # 创建必要的目录
    for d in [folder_1, folder_2]:
        if not os.path.exists(d):
            os.makedirs(d)

    json_files = [f for f in os.listdir(target_dir) if f.lower().endswith('.json')]

    for json_name in json_files:
        json_path = os.path.join(target_dir, json_name)
        img_basename = os.path.splitext(json_name)[0]
        
        # 寻找对应的图片
        img_name = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            if os.path.exists(os.path.join(target_dir, img_basename + ext)):
                img_name = img_basename + ext
                break
        
        if not img_name:
            continue

        img_path = os.path.join(target_dir, img_name)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        shapes = data.get('shapes', [])
        
        # --- 步骤 1: 检查是否存在 3 号类 ---
        # 只要有一个 label 为 "3" 
        has_class_3 = any(str(s['label']) == "3" for s in shapes)

        # --- 步骤 2: 处理/过滤 shapes ---
        new_shapes = []
        for s in shapes:
            label_str = str(s['label'])
            
            # 去除 1 号类
            if label_str == "1":
                continue
            
            # 如果存在 3 号类，则把 2 和 3 都转为 0
            if has_class_3 and (label_str == "2" or label_str == "3"):
                s['label'] = "0"
                new_shapes.append(s)
            # 如果不存在 3 号类，但原本有 2 号类，这里保留 2（或根据需要处理）
            # 按你要求：逻辑核心在于有 3 则转 2,3 为 0
            elif label_str == "2":
                s['label'] = "0" # 统一将剩余的2也转为0以便后续使用
                new_shapes.append(s)
            elif label_str == "0":
                new_shapes.append(s)

        # --- 步骤 3: 分流保存 ---
        # A. 如果处理后 shapes 为空
        if not new_shapes:
            os.remove(json_path)
            shutil.move(img_path, os.path.join(folder_2, img_name))
            print(f"🗑️ [空] 已移除并移至 2号文件夹: {img_name}")
        
        # B. 如果处理后不为空，且曾经包含 3 号类 (迁往 1 号文件夹)
        elif has_class_3:
            data['shapes'] = new_shapes
            # 保存到 1 号文件夹
            with open(os.path.join(folder_1, json_name), 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            shutil.move(img_path, os.path.join(folder_1, img_name))
            # 同时也删除原位置的 json（因为图片已经 move 走了）
            os.remove(json_path)
            print(f"🚀 [含有3] 已转换并迁至 1号文件夹: {img_name}")
            
        # C. 如果不含 3 号类但也非空 (原地保留或按需处理)
        else:
            data['shapes'] = new_shapes
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ [常规] 已原地更新 JSON: {json_name}")

    print("\n--- 处理任务结束 ---")

# --- 参数设置 ---
refine_labels_logic(
    target_dir='123', # 源文件夹
    folder_1='tmp',     # 1号文件夹 (含3类的处理结果)
    folder_2='special_background'
)
