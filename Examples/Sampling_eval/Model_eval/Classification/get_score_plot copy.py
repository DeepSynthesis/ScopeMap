import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame([], columns=['index', '20 samples', '40 samples', '60 samples', '80 samples', '100 samples'])

df['index'] = ['Accuracy', 'Recall', 'F1']
df['20 samples'] = [0.8588, 1.0000, 0.9240]
df['40 samples'] = [0.8636, 0.9834, 0.9253]
df['60 samples'] = [0.8461, 0.9661, 0.9151]
df['80 samples'] = [0.8541, 0.9957, 0.9212]
df['100 samples'] = [0.8512, 0.9985, 0.9192]

acc_df = pd.DataFrame([], columns=['index', 'values', 'class'])
acc_df['index'] = ['20 samples', '40 samples', '60 samples', '80 samples', '100 samples']
acc_df['values'] = [0.8588, 0.8636, 0.8461, 0.8541, 0.8512]
acc_df['class'] = ['Accuracy', 'Accuracy', 'Accuracy', 'Accuracy', 'Accuracy']

re_df = pd.DataFrame([], columns=['index', 'values', 'class'])
re_df['index'] = ['20 samples', '40 samples', '60 samples', '80 samples', '100 samples']
re_df['values'] = [1.0000, 0.9834, 0.9661, 0.9957, 0.9985]
re_df['class'] = ['Recall', 'Recall', 'Recall', 'Recall', 'Recall']

f1_df = pd.DataFrame([], columns=['index', 'values', 'class'])
f1_df['index'] = ['20 samples', '40 samples', '60 samples', '80 samples', '100 samples']
f1_df['values'] = [0.9240, 0.9253, 0.9151, 0.9212, 0.9192]
f1_df['class'] = ['F1', 'F1', 'F1', 'F1', 'F1']
df = pd.concat([acc_df, re_df, f1_df], axis=0)

plt.figure(figsize=(10, 8))
sns.lineplot(data=df, x='index', y='values', hue='class', marker='o')
plt.ylim(0.6, 1.0)
plt.legend(loc='lower right', fontsize=18)
plt.xlabel('Number of samples', fontsize=20)
plt.ylabel('Score', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
plt.savefig('get_score_plot.png')
plt.savefig('get_score_plot.svg')