import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ========== 全局字体设置 ==========
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 20, 'axes.titlesize': 24})

# # ========== 数据与坐标转换 ==========构件时间
#x_values = np.array([500, 10, 1])
#y_values = np.array([384,168, 36])
#z_matrix = np.array([[3.8,2.6,1],
#                      [0.33,0.025,0.026],

# [0.006,0.004,0.003]
#                      ])

# ========== 数据与坐标转换 ==========构件内存
#x_values = np.array([500, 10, 1])
#y_values = np.array([384,168, 36])
#z_matrix = np.array([
#    [174,146,144],
#    [180,145,143],
#    [180,150,143]
#                     ])

# # # ========== 数据与坐标转换 ==========构件组时间
x_values = np.array([217, 10, 2])
y_values = np.array([10,3, 2])
z_matrix = np.array([
     [8,6,4.24],
     [1.5,0.86,1.05],
     [1.26,0.88,0.85]
                      ])

# # ========== 数据与坐标转换 ==========构件组内存
# x_values = np.array([217, 10, 3])
# y_values = np.array([10,3, 2])
# z_matrix = np.array([
#     [307,278,279],
#     [203,198,199],
#     [200,197,199]
#                      ])


x_indices = np.arange(len(x_values))
y_indices = np.arange(len(y_values))
xpos, ypos = np.meshgrid(x_indices, y_indices, indexing='ij')

BAR_WIDTH = 0.4
dx = dy = BAR_WIDTH
dz = z_matrix.ravel()
xpos = xpos.ravel() - dx/2
ypos = ypos.ravel() - dy/2
zpos = np.zeros_like(xpos)

# ========== 绘图设置 ==========
fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.coolwarm(dz / dz.max())

ax.bar3d(xpos, ypos, zpos, dx, dy, dz,
         color=colors, edgecolor='k', alpha=0.8)

# # ========== 坐标轴与标签设置 ==========构件时间
# ax.set_xlabel('构件数量', fontsize=24, labelpad=15)
# ax.set_ylabel('面片数量', fontsize=24, labelpad=15)
# ax.set_zlabel('耗时(s)', fontsize=24, labelpad=15)

# ========== 坐标轴与标签设置 ==========构件空间
ax.set_xlabel('Components to be retrieved', fontsize=24, labelpad=15)
ax.set_ylabel('Number of components in group', fontsize=24, labelpad=15)
ax.set_zlabel('Time Consuming(s)', fontsize=24, labelpad=15)

# # # ========== 坐标轴与标签设置 ==========构件组时间
# ax.set_xlabel('待比对模型构件数量', fontsize=24, labelpad=15)
# ax.set_ylabel('构件组构件数量', fontsize=24, labelpad=15)
# ax.set_zlabel('耗时(s)', fontsize=24, labelpad=15)

# ========== 坐标轴与标签设置 ==========构件组空间
# ax.set_xlabel('待比对模型构件数量', fontsize=24, labelpad=15)
# ax.set_ylabel('构件组构件数量', fontsize=24, labelpad=15)
# ax.set_zlabel('占用空间(MB)', fontsize=24, labelpad=15)
#
ax.set_xticks(x_indices)
ax.set_xticklabels(x_values, fontsize=20)
ax.set_yticks(y_indices)
ax.set_yticklabels(y_values, fontsize=20)
ax.tick_params(axis='z', labelsize=20)

# ========== 颜色条设置 ==========
from matplotlib.colors import Normalize
mappable = plt.cm.ScalarMappable(
    cmap='coolwarm',
    norm=Normalize(vmin=0, vmax=1)  # 强制设置范围 [0, 200]
)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.5)
cbar.set_label('Time Consuming', rotation=270, labelpad=25, fontsize=24)
#cbar.set_label('占用空间', rotation=270, labelpad=25, fontsize=24)
cbar.set_ticks([0,0.2, 0.4,0.6,0.8,1.0])  # 手动指定刻度位置

# cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.coolwarm), ax=ax, shrink=0.5)
# # cbar.set_label('时间强度', rotation=270, labelpad=25, fontsize=24)
# cbar.set_label('占用空间', rotation=270, labelpad=25, fontsize=24)
cbar.ax.tick_params(labelsize=20)

# ========== 视角与布局 ==========
ax.view_init(elev=30, azim=45)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)  # 手动调整边距

# ========== 保存与显示 ==========
output_path = r"F:\OneDrive\桌面\绘图代码\构件组时间图.png"
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 确保目录存在

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.5,
    facecolor='white'
)

plt.show()