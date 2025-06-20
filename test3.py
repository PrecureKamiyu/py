import numpy as np

num_points = 20
x_min, x_max = 0, 1
y_min, y_max = 0, 1

x_coords = np.round(np.random.uniform(x_min, x_max, num_points), 3)
y_coords = np.round(np.random.uniform(y_min, y_max, num_points), 3)
points = np.column_stack((x_coords, y_coords))

print(f"生成了 {num_points} 个随机点 (NumPy 数组，小数点后三位):")
print(f"X 坐标前5个: {x_coords[:5]}")
print(f"Y 坐标前5个: {y_coords[:5]}")
print(f"组合点前5个:\n{points[:5]}") # 打印组合后的点
