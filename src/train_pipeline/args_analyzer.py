import yaml
import numpy as np
import matplotlib.pyplot as plt
import math

# 解决 Matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def plot_changed_hyperparameters(default_yaml_path, best_yaml_path):
    # 1. 读取数据
    print("正在读取超参数文件...")
    try:
        default_hyp = load_yaml(default_yaml_path)
        best_hyp = load_yaml(best_yaml_path)
    except FileNotFoundError as e:
        print(f"文件未找到，请检查路径！\n{e}")
        return

    # 2. 提取交集中的数值型参数，且只保留“改变项”
    keys = []
    defaults = []
    bests = []
    
    for k, best_val in best_hyp.items():
        if k in default_hyp:
            default_val = default_hyp[k]
            # 只处理数字类型的参数（排除 string, bool 等）
            if isinstance(best_val, (int, float)) and isinstance(default_val, (int, float)):
                d_val = float(default_val)
                b_val = float(best_val)
                
                # 核心过滤逻辑：使用 isclose 忽略极微小的浮点数精度误差
                # 只有当相对差异大于万分之一，或绝对差异大于百万分之一时，才认定为“改变项”
                if not math.isclose(d_val, b_val, rel_tol=1e-4, abs_tol=1e-6):
                    keys.append(k)
                    defaults.append(d_val)
                    bests.append(b_val)

    if not keys:
        print("对比完成：最佳超参数与默认参数完全一致，没有任何数值改变！")
        return

    print(f"发现 {len(keys)} 项超参数发生了改变，正在生成可视化图表...")

    # 3. 计算相对变化率 (相对于 default 的百分比)
    changes = []
    for d, b in zip(defaults, bests):
        if d == 0:
            change = (b - d) / 1e-5 * 100 if b != 0 else 0
        else:
            change = ((b - d) / abs(d)) * 100
        changes.append(change)

    # ================= 绘图部分 =================
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f'YOLO 进化前后超参数对比分析 (仅展示改变的 {len(keys)} 项)', fontsize=16, fontweight='bold')

    # ------ 子图 1：百分比变化水平柱状图 ------
    ax1 = fig.add_subplot(121)
    
    colors = ['#ff9999' if c > 0 else '#99cc99' for c in changes]
    
    y_pos = np.arange(len(keys))
    bars = ax1.barh(y_pos, changes, color=colors, edgecolor='grey')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(keys, fontsize=10)
    ax1.invert_yaxis()  
    ax1.set_xlabel('相对默认参数的变化百分比 (%)')
    ax1.set_title('超参数增减幅度 (条形图)')
    ax1.grid(axis='x', linestyle='--', alpha=0.7)

    for bar, change in zip(bars, changes):
        xval = bar.get_width()
        label_x_pos = xval + 2 if xval > 0 else xval - 2
        ha = 'left' if xval > 0 else 'right'
        display_val = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
        if abs(change) > 1000: display_val = ">1000%"
        ax1.text(label_x_pos, bar.get_y() + bar.get_height()/2, display_val, 
                 va='center', ha=ha, fontsize=8)

    # ------ 子图 2：参数向量雷达图 (归一化) ------
    ax2 = fig.add_subplot(122, polar=True)
    
    N = len(keys)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1] 

    norm_defaults = [1.0] * N
    norm_bests = []
    for d, b in zip(defaults, bests):
        if d == 0:
            norm_bests.append(b + 1.0) 
        else:
            ratio = b / d
            norm_bests.append(min(ratio, 3.0) if ratio > 0 else max(ratio, -1.0))
            
    norm_defaults += norm_defaults[:1] 
    norm_bests += norm_bests[:1]       

    ax2.plot(angles, norm_defaults, linewidth=2, linestyle='solid', label='Default (基准)', color='blue')
    ax2.fill(angles, norm_defaults, 'blue', alpha=0.1)

    ax2.plot(angles, norm_bests, linewidth=2, linestyle='solid', label='Best (调优后)', color='red')
    ax2.fill(angles, norm_bests, 'red', alpha=0.25)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(keys, size=9)
    ax2.set_yticklabels([]) 
    ax2.set_title('改变项的参数向量偏移 (雷达图)\n基准环=Default参数')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout()
    
    plt.savefig('changed_hyperparameters_comparison.png', dpi=300, bbox_inches='tight')
    print("图表已生成并保存为 'changed_hyperparameters_comparison.png'")
    plt.show()

if __name__ == "__main__":
    # 配置你的 YAML 文件路径
    DEFAULT_YAML = "data/args/args.yaml" 
    BEST_YAML = "data/args/best.yaml" 
    
    plot_changed_hyperparameters(DEFAULT_YAML, BEST_YAML)

