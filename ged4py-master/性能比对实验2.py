# #构件级时间，自变量面片数量
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
# plt.rcParams['axes.unicode_minus'] = False
#
# # 参数设置
# e = np.arange(1, 1000)       # 构件数量范围（1-249）
# y = 1                # 剪枝率
# c = 1         # 单次匹配时间（ms）
#
# # 定义实际时间数据的二维矩阵：每行表示一个点的坐标 [构件数量, 实际时间]
# actual_matrix = np.array([
#     [0, 0],
#     [200,36], [400, 146], [600, 450],
#     [800, 605], [1000, 954]
# ])
#
# # 提取矩阵中的横纵坐标
# x_actual = actual_matrix[:, 0]  # 第一列为构件数量
# y_actual = actual_matrix[:, 1]  # 第二列为实际时间
#
# # 计算理论时间复杂度曲线
# theory_time = ( e) **2*0.0007
#
# # 创建画布
# plt.figure(figsize=(8, 6), facecolor='#F5F5F5')
#
# # 绘制实际数据折线（从矩阵中提取坐标点并连线）
# plt.plot(x_actual, y_actual,
#          marker='o',
#          linestyle='--',
#          color='#4169E1',
#          linewidth=2,
#          markersize=8,
#          label='实际耗时')
#
# # 绘制理论复杂度曲线
# plt.plot(e, theory_time,
#          marker='',
#          linestyle='-',
#          color='#FF4500',
#          linewidth=3,
#          label=r'理论复杂度')
#
# # 设置坐标轴
# plt.xlabel("待匹配模型构件数量", fontsize=18)
# plt.ylabel("时间 (s)", fontsize=18)
# plt.xticks(np.arange(0, 1001, 200))  # 调整刻度范围匹配实际数据
# plt.yticks(np.arange(0, 1001, 200))  # 调整刻度范围匹配实际数据
# plt.grid(True, linestyle='--', alpha=0.6)
#
# # 添加图例与标题
# plt.legend(loc='upper left', fontsize=18)











# #构件级时间，自变量面片数量
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
# plt.rcParams['axes.unicode_minus'] = False
#
# # 参数设置
# e = np.arange(2, 11)       # 构件数量范围（1-249）
# y = 1                # 剪枝率
# c = 8          # 单次匹配时间（ms）
#
# # 定义实际时间数据的二维矩阵：每行表示一个点的坐标 [构件数量, 实际时间]
# actual_matrix = np.array([
#      [2,36], [4, 39], [6, 41],
#     [8, 43], [10, 46]
# ])
#
# # 提取矩阵中的横纵坐标
# x_actual = actual_matrix[:, 0]  # 第一列为构件数量
# y_actual = actual_matrix[:, 1]  # 第二列为实际时间
#
# # 计算理论时间复杂度曲线
# theory_time = (y * e)+34
#
# # 创建画布
# plt.figure(figsize=(8, 6), facecolor='#F5F5F5')
#
# # 绘制实际数据折线（从矩阵中提取坐标点并连线）
# plt.plot(x_actual, y_actual,
#          marker='o',
#          linestyle='--',
#          color='#4169E1',
#          linewidth=2,
#          markersize=8,
#          label='实际耗时')
#
# # 绘制理论复杂度曲线
# plt.plot(e, theory_time,
#          marker='',
#          linestyle='-',
#          color='#FF4500',
#          linewidth=3,
#          label=r'理论复杂度 ')
#
# # 设置坐标轴
# plt.xlabel("构件组构件数量", fontsize=18)
# plt.ylabel("时间 (s)", fontsize=18)
# plt.xticks(np.arange(0, 11, 2))  # 调整刻度范围匹配实际数据
# plt.grid(True, linestyle='--', alpha=0.6)
#
# # 添加图例与标题
# plt.legend(loc='upper left', fontsize=18)




#
# #构件级空间，自变量构件面片数量
# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
# plt.rcParams['axes.unicode_minus'] = False
#
# # 新参数设置
# n= 240                  # 固定构件数量
# k = 16                  # 剪枝率
# m = np.arange(1, 1000)
#
# # 实际时间数据矩阵（需调整为[c值, 实际时间]）
# actual_matrix = np.array([
#      [0,0],[200,842], [400, 2749], [600, 5882],
#     [800, 10302], [1000, 15965]
# ])
# # 提取坐标
# x_actual = actual_matrix[:, 0]  # 第一列为c值
# y_actual = actual_matrix[:, 1]  # 第二列为实际时间
#
# # 计算理论曲线（基于固定e值和变化的c）
# theory_time = m**2*0.01
#
# # 创建画布
# plt.figure(figsize=(8, 6), facecolor='#F5F5F5')
#
# # 绘制实际数据（x轴为c值）
# plt.plot(x_actual, y_actual,
#          marker='o',
#          linestyle='--',
#          color='#4169E1',
#          linewidth=2,
#          markersize=8,
#          label='实际空间占用')
#
# # 绘制理论曲线（x轴为c值）
# plt.plot(m, theory_time,
#          marker='',
#          linestyle='-',
#          color='#FF4500',
#          linewidth=3,
#          label=r'理论复杂度')
#
# # 设置坐标轴
# plt.xlabel("待匹配模型构件数量 ", fontsize=18)
# plt.ylabel("占用空间（MB）", fontsize=18)
# plt.xticks(np.arange(0, 1001, 200))  # 调整刻度范围匹配实际数据
# plt.grid(True, linestyle='--', alpha=0.6)
#
# # 添加图例与标题
# plt.legend(loc='upper left', fontsize=18)
#
#




#构件级空间，自变量构件面片数量
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False

# 新参数设置
n= 240                  # 固定构件数量
k = 2                    # 剪枝率
m = np.arange(2, 11)

# 实际时间数据矩阵（需调整为[c值, 实际时间]）
actual_matrix = np.array([
     [2,842], [4, 844], [6, 847],
    [8, 849], [10, 850]
])
# 提取坐标
x_actual = actual_matrix[:, 0]  # 第一列为c值
y_actual = actual_matrix[:, 1]  # 第二列为实际时间

# 计算理论曲线（基于固定e值和变化的c）
theory_time = m*0+842

# 创建画布
plt.figure(figsize=(8, 6), facecolor='#F5F5F5')

# 绘制实际数据（x轴为c值）
plt.plot(x_actual, y_actual,
         marker='o',
         linestyle='--',
         color='#4169E1',
         linewidth=2,
         markersize=8,
         label='实际空间占用')

# 绘制理论曲线（x轴为c值）
plt.plot(m, theory_time,
         marker='',
         linestyle='-',
         color='#FF4500',
         linewidth=3,
         label=r'理论复杂度')

# 设置坐标轴
plt.xlabel("构件组构件数量 ", fontsize=18)
plt.ylabel("占用空间（MB）", fontsize=18)
plt.xticks(np.arange(0, 11, 2))  # 调整刻度范围匹配实际数据
plt.yticks(np.arange(0, 1201, 200))  # 调整刻度范围匹配实际数据
plt.grid(True, linestyle='--', alpha=0.6)

# 添加图例与标题
plt.legend(loc='upper left', fontsize=18)





# ========== 保存与显示 ==========
output_path = r"C:\Users\admin\Desktop\蔡建华毕业论文\性能图4.png"
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