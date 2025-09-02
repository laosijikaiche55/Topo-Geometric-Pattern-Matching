import matplotlib.pyplot as plt
import numpy as np

# ========== 全局字体放大设置 ==========
plt.rcParams.update({
    'font.size': 24,          # 基础字体放大一倍（原12→24）
    'axes.titlesize': 28,     # 标题字体
    'axes.labelsize': 24,     # 坐标轴标签
    'xtick.labelsize': 24,    # X轴刻度
    'ytick.labelsize': 24,    # Y轴刻度
    'legend.fontsize': 20,    # 图例字体
    'font.sans-serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black'
})

# 生成数据
delta_L = [0.0001, 0.0002, 0.0005, 0.001,0.002, 0.005,0.01, 0.02, 0.05, 0.1]
tpr = [0, 0, 0.2, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0]
#tpr = [0, 0.7, 0.9, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0]
# tpr = [0, 0, 0, 0, 0, 0,0, 0, 0, 0]
# tpr = [0, 0, 0, 0, 0, 0,0, 0, 0, 0]

# ========== 画布尺寸优化 ==========
fig, ax1 = plt.subplots(figsize=(12, 7.2), dpi=100)  # 尺寸放大1.2倍（原10x6→12x7.2）

# 绘制TPR曲线
line = ax1.plot(delta_L, tpr, color='#2A6BBD', linestyle='-',
                linewidth=3, marker='o', markersize=12,  # 线宽和标记放大
                markerfacecolor='white', markeredgewidth=2)

# ========== 坐标轴设置 ==========
ax1.set_ylabel('True Positive Rate', color='black', labelpad=20)  # labelpad增加33%
# ax1.set_ylabel('假阳性率 (FPR)', color='black', labelpad=20)  # labelpad增加33%
ax1.tick_params(axis='y', length=8, width=1.5)  # 刻度线同步放大
ax1.set_xscale('log')
ax1.set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
ax1.set_xticklabels(['0.0001', '0.001', '0.01', '0.1'],
                   fontdict={'fontsize':24})  # 显式设置刻度标签
ax1.grid(True, linestyle=':', color='gray', alpha=0.6)

# ========== 图例强化 ==========
leg = ax1.legend(['True Positive Rate'], loc='lower right', frameon=True,
                framealpha=0.9, edgecolor='white', handlelength=1.5)
# leg = ax1.legend(['假阳性率'], loc='lower right', frameon=True,
#                 framealpha=0.9, edgecolor='white', handlelength=1.5)
for text in leg.get_texts():
    text.set_fontweight('semibold')  # 增强可读性

# ========== 导出优化 ==========
plt.tight_layout(pad=3)  # 布局边距增加
plt.savefig('尺寸差异检测曲线.png', format='png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()