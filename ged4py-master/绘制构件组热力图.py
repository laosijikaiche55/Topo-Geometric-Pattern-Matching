import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# === 1. 自定义蓝白渐变色系 ===
colors = ["#FFFFFF", "#6BAED6", "#2171B5", "#08306B"]
cmap = LinearSegmentedColormap.from_list("blue_white", colors)

# === 2. 7x7矩阵数据 ===
# 构件组名称（7个示例）
component_groups = [" ", " ", " ", " ", " ", " ", " "]

# 预定义7x7相似度矩阵（可在此处替换实际数据）
similarity_matrix = np.array([
    [0.20, 0.20, 0.47, 0.57, 0.47, 0.70, 1.00],
    [0.20, 0.20, 0.40, 0.47, 0.57, 1.00, 0.70],
    [0.20, 0.20, 0.47, 0.57, 1.00, 0.57, 0.47],
    [0.40, 0.47, 0.80, 1.00, 0.57, 0.47, 0.57],
    [0.58, 0.68, 1.00, 0.80, 0.47, 0.40, 0.47],
    [0.75, 1.00, 0.68, 0.47, 0.20, 0.20, 0.20],
    [1.00, 0.75, 0.58, 0.40, 0.20, 0.20, 0.20]
])

# === 3. 可视化配置 ===
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 创建DataFrame
df = pd.DataFrame(similarity_matrix,
                 index=component_groups,
                 columns=component_groups)

# === 4. 绘制7x7热力图 ===
plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    df,
    annot=False,  # 关闭默认注释
    fmt=".2f",
    cmap=cmap,
    vmin=0,
    vmax=1,
    linewidths=0.8,
    linecolor='white',
    cbar_kws={
        "label": "相似度系数",
        "ticks": np.linspace(0, 1, 11)
    }
)

# === 新增：动态设置字体颜色 ===
for i in range(len(component_groups)):
    for j in range(len(component_groups)):
        value = similarity_matrix[i, j]
        color = 'white' if value > 0.8 else 'black'  # 阈值判断
        ax.text(j + 0.5, i + 0.5,                 # 坐标位置修正
                f'{value:.2f}',                     # 保留两位小数
                ha='center', va='center',          # 居中显示
                fontsize=18,
                color=color,
                fontweight='semibold')              # 保持原有字体粗细

# === 5. 增强可视化效果 ===
# 调整坐标轴标签
# ax.set_title('7x7 构件组相似度热力图', fontsize=18, pad=25)
ax.set_xticklabels(ax.get_xticklabels(),
                  rotation=45,
                  ha='right',
                  fontsize=12,
                  fontweight='semibold')  # 加粗字体

ax.set_yticklabels(ax.get_yticklabels(),
                  fontsize=12,
                  fontweight='semibold')

# 增强颜色条
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
cbar.ax.set_ylabel('相似度等级',
                  rotation=-90,
                  va="bottom",
                  fontsize=12,
                  fontweight='bold')

# 添加自定义网格
ax.hlines([i for i in range(1,7)], *ax.get_xlim(), colors='white', linewidths=0.8)
ax.vlines([i for i in range(1,7)], *ax.get_ylim(), colors='white', linewidths=0.8)

# === 6. 导出与显示 ===
plt.tight_layout()
plt.savefig('7x7_构件组相似度热力图.png', dpi=300, bbox_inches='tight')
plt.show()