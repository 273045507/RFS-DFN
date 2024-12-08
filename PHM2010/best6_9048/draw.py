import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('line.csv')

# 绘制训练和验证损失
plt.figure(figsize=(14, 8))

# 绘制训练损失
plt.plot(data['epoch'], data['Train Loss'], label='Train Loss', color='blue', linestyle='-', marker='o')

# 绘制验证损失
plt.plot(data['epoch'], data['Val Loss'], label='Val Loss', color='red', linestyle='-', marker='x')

# 绘制训练准确率
plt.plot(data['epoch'], data['Train Accuracy'], label='Train Accuracy', color='green', linestyle='--', marker='o')

# 绘制验证准确率
plt.plot(data['epoch'], data['Val Accuracy'], label='Val Accuracy', color='orange', linestyle='--', marker='x')

# 设置横坐标的刻度间隔为1
plt.xticks(range(int(data['epoch'].min()), int(data['epoch'].max()) + 1, 1))

# 设置坐标轴刻度字体大小
plt.tick_params(axis='both', which='major', labelsize=16)

# 设置标题和标签的字体大小
plt.title('(f) Training dominated by Z-axis vibration', fontsize=20)
# plt.xlabel('Epoch', fontsize=18)
# plt.ylabel('Value', fontsize=18)

# 设置图例放在下方，横向显示，字体大小
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=15)

# 显示图形
plt.grid(True)
plt.show()

