import matplotlib.pyplot as plt
import numpy as np

with open('logs/train2024-04-25-145015.log', 'r') as f:
    data = f.readlines()

data = [i.split() for i in data[8:]]
data = np.array(data)
loss = [float(i[-1]) for i in data]

# 将列表 "loss" 转换为NumPy数组
loss = np.array(loss)

degree = 10

# 进行拟合
x = np.arange(len(loss))  # 使用数据点的索引作为x值
coefficients = np.polyfit(x, loss, degree)
poly_function = np.poly1d(coefficients)

# 绘制原始数据
plt.plot(x, loss, label='Original data')

# 绘制拟合曲线
x_values = np.linspace(0, len(loss), 100)
plt.plot(x_values, poly_function(x_values), color='red', label='Fitted line')

plt.xlabel('Index')
plt.ylabel('Loss')
plt.title('Curve Fitting with Matplotlib')
plt.legend()
plt.grid(True)
plt.show()