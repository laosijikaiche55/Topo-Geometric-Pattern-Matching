import numpy as np
import sympy as sp

# 定义两个矩阵
A = np.array([[1, 2.00001, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])

# 将NumPy数组转换为SymPy矩阵
A_sym = sp.Matrix(A)
B_sym = sp.Matrix(B)

# 计算行最简形式
RREF_A, _ = A_sym.rref()
RREF_B, _ = B_sym.rref()

# 将行最简形式转换为NumPy数组
RREF_A_np = np.array(RREF_A).astype(float)
RREF_B_np = np.array(RREF_B).astype(float)

# 设置容差
tolerance = 1e-5

# 检查行最简形式是否在容差范围内近似相等
are_equivalent = np.allclose(RREF_A_np, RREF_B_np, atol=tolerance)
print("矩阵是否在容差范围内等价：", are_equivalent)
