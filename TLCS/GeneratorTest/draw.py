import pandas as pd
import matplotlib.pyplot as plt

# 读取csv文件
df = pd.read_csv('data.csv')

# 重置索引为整数序列
df = df.reset_index()

# 绘制第一列的折线图
plt.plot(df.index, df.iloc[:, 1])
plt.xlabel('Index')
plt.ylabel('y1')
plt.title('Line chart for y1')
plt.show()

# 绘制第二列的折线图
plt.plot(df.index, df.iloc[:, 2])
plt.xlabel('Index')
plt.ylabel('y2')
plt.title('Line chart for y2')
plt.show()

# 绘制第三列的折线图
plt.plot(df.index, df.iloc[:, 3])
plt.xlabel('Index')
plt.ylabel('y3')
plt.title('Line chart for y3')
plt.show()
