import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# 读取cvt.log文件
with open('Classification.log', 'r') as f:
    content = f.read()

# 解析Accuracy, Recall, F1 Score
pattern = r'Accuracy: ([\d.]+)\s+Recall: ([\d.]+)\s+F1 Score: ([\d.]+)'
matches = re.findall(pattern, content)

# 提取数据
accuracys = [float(m[0]) for m in matches]
recalls = [float(m[1]) for m in matches]
f1s = [float(m[2]) for m in matches]

print(f"找到 {len(accuracys)} 组数据")

# 样本数对应关系
sample_sizes = [20, 40, 60, 80, 100]
n_splits = 5

# 按样本数分组计算均值和标准差
acc_means, acc_stds = [], []
rec_means, rec_stds = [], []
f1_means, f1_stds = [], []

for i, size in enumerate(sample_sizes):
    start_idx = i * n_splits
    end_idx = start_idx + n_splits

    acc_means.append(np.mean(accuracys[start_idx:end_idx]))
    acc_stds.append(np.std(accuracys[start_idx:end_idx]))
    rec_means.append(np.mean(recalls[start_idx:end_idx]))
    rec_stds.append(np.std(recalls[start_idx:end_idx]))
    f1_means.append(np.mean(f1s[start_idx:end_idx]))
    f1_stds.append(np.std(f1s[start_idx:end_idx]))

    print(f"{size} samples: Acc={np.mean(accuracys[start_idx:end_idx]):.4f}±{np.std(accuracys[start_idx:end_idx]):.4f}, Recall={np.mean(recalls[start_idx:end_idx]):.4f}±{np.std(recalls[start_idx:end_idx]):.4f}, F1={np.mean(f1s[start_idx:end_idx]):.4f}±{np.std(f1s[start_idx:end_idx]):.4f}")

# Set up bar positions
samples = ['20', '40', '60', '80', '100']
x = np.arange(len(samples))
width = 0.2

# Create figure and axis
plt.figure(figsize=(12, 8))

# Create grouped bars with error bars
bars1 = plt.bar(x - width, acc_means, width, label='Accuracy', color='#D87792', edgecolor='#CA486B', linewidth=2, yerr=acc_stds, capsize=4)
bars2 = plt.bar(x, rec_means, width, label='Recall', color='#2C75E3', edgecolor='#172C51', linewidth=2, yerr=rec_stds, capsize=4)
bars3 = plt.bar(x + width, f1_means, width, label='F1', color='#29ABBA', edgecolor='#228E9B', linewidth=2, yerr=f1_stds, capsize=4)

# Customize the plot
plt.xlabel('Number of samples', fontsize=30)
plt.ylabel('Score', fontsize=30)
plt.ylim(0.60, 1.05)
plt.xticks(x, samples, fontsize=27)
plt.yticks(fontsize=27)
plt.legend(loc='lower right', fontsize=27)

plt.tight_layout()
plt.savefig('get_score_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('get_score_plot.svg', bbox_inches='tight')
plt.show()
