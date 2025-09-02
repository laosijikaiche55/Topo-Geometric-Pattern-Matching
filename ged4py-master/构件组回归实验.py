import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（确保系统已安装对应字体）
plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 生成模拟数据
delta_L = np.logspace(-4, -1, num=10, base=10)  # 10^[-4, -1]
tpr = [0, 0, 0.2, 0.65,0.92, 0.94, 1.0, 1.0, 1.0, 1.0]  # 真阳性率
fpr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 假阳性率

# 创建画布和坐标轴
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=100)

# 绘制TPR曲线（主坐标轴）
ax1.plot(delta_L, tpr,
         color='#2A6BBD',  # 使用HEX色值
         linestyle='-',
         linewidth=2,
         marker='o',
         markersize=8,
         markerfacecolor='white',
         markeredgewidth=1.5,
         label='真阳性率 (δ=0.5mm)')

# 设置主坐标轴属性
ax1.set_xlabel('尺寸差异 (ΔL, 毫米)',
              fontsize=13,
              labelpad=10)
ax1.set_ylabel('真阳性率 (TPR)',
              color='#2A6BBD',
              fontsize=13,
              labelpad=10)
ax1.tick_params(axis='y', labelcolor='#2A6BBD')
ax1.set_xscale('log')  # 设置对数坐标轴
ax1.set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
ax1.set_xticklabels(['0.0001', '0.001', '0.01', '0.1'])
ax1.grid(True,
        linestyle=':',
        color='gray',
        alpha=0.6)

# 创建次坐标轴
ax2 = ax1.twinx()

# 绘制FPR曲线（次坐标轴）
ax2.plot(delta_L, fpr,
         color='#D95319',  # 使用HEX色值
         linestyle='--',
         linewidth=2,
         marker='s',
         markersize=8,
         markerfacecolor='white',
         markeredgewidth=1.5,
         label='假阳性率 (δ=0.5mm)')

# 设置次坐标轴属性
ax2.set_ylabel('假阳性率 (FPR)',
              color='#D95319',
              fontsize=13,
              labelpad=10)
ax2.tick_params(axis='y', labelcolor='#D95319')

# 添加标题和图例
plt.title('尺寸差异对检测性能的影响',
         fontsize=15,
         pad=15)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
          loc='lower right',
          frameon=True,
          framealpha=0.9,
          edgecolor='white')

# 优化布局并保存
plt.tight_layout()

# 导出PNG格式（支持透明背景）
plt.savefig('尺寸差异检测曲线.png',
           format='png',
           dpi=300,
           bbox_inches='tight',
           transparent=False,  # 可设置为True启用透明背景
           facecolor='white')  # 背景颜色设置

plt.show()