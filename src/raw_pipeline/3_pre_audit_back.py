###  0 原先类别  1预标注类别  其余：预标注错误类别，手动修改
import os
import shutil

# ================= 配置区域 =================
labels_dir = "data\\2_deduplicated_image\\labels"       # 原始 txt 标签路径
images_dir = "data\\2_deduplicated_image\\kept"        # 原始图片路径

folder_1 = "data/4_raw_train/image"       # 1号文件夹：存放处理后有标签的图片
folder_2 = "data/4_raw_train/label"       # 2号文件夹：存放处理后的 txt 标签
folder_1_1 = "data/past_raw_train/image"
folder_2_1 = "data/past_raw_train/label"
folder_bg = "data/special_background"     # 特殊背景：仅含类别2,3的图片
folder_others = "data/past_background"     # 3号文件夹：无标签图片、纯空标签图片等

img_ext = ".jpg" # 确保与你的图片后缀一致
# ============================================

# 确保目标文件夹存在
for f in [folder_1, folder_2, folder_bg, folder_others,folder_1_1,folder_2_1]:
    os.makedirs(f, exist_ok=True)

# ⭐️ 核心改动：改为遍历所有【图片】文件
for img_name in os.listdir(images_dir):
    if not img_name.endswith(img_ext):
        continue

    img_path = os.path.join(images_dir, img_name)
    base_name = os.path.splitext(img_name)[0]
    txt_name = base_name + ".txt"
    txt_path = os.path.join(labels_dir, txt_name)

    # 检查该图片是否有对应的 txt 标签
    if os.path.exists(txt_path):
        # ========== 有标签的情况 ==========
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        original_classes = set()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            cls_id = int(parts[0])
            original_classes.add(cls_id)

            if cls_id == 1:
                parts[0] = '0'
                new_lines.append(' '.join(parts) + '\n')
            elif cls_id in [2, 3]:
                continue
            else:
                new_lines.append(line)

        # 判断复制去向
        if len(new_lines) > 0:
            # 【情况 A】 处理后标签不为空 -> 1号和2号文件夹
            new_txt_dest = os.path.join(folder_2, txt_name)
            with open(new_txt_dest, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            new_txt_dest = os.path.join(folder_2_1, txt_name)
            with open(new_txt_dest, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            shutil.copy2(img_path, os.path.join(folder_1_1, img_name)) #复制一份到长期文件夹
            shutil.copy2(img_path, os.path.join(folder_1, img_name))  

        elif len(original_classes) > 0 and original_classes.issubset({2, 3}):
            # 【情况 B】 原本只有类别 2 和 3 -> 特殊背景文件夹
            shutil.copy2(img_path, os.path.join(folder_bg, img_name))

        else:
            # 【情况 C】 有 txt 但为空，或其他情况 -> 3号文件夹
            shutil.copy2(img_path, os.path.join(folder_others, img_name))

    else:
        # ========== 无标签的情况 ==========
        # 【情况 D】 找不到对应的 txt 文件 -> 直接归入 3号文件夹
        shutil.copy2(img_path, os.path.join(folder_others, img_name))

print("✅ 数据集清洗与转移完成！所有图片均已归档，原始数据已保留。")
