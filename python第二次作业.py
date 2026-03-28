import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

#  1. 数据准备
# 真实标签（one-hot形式）
y_true = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 0]
])

# 预测概率
y_score = np.array([
    [0.1, 0.2, 0.7],
    [0.1, 0.6, 0.3],
    [0.5, 0.2, 0.3],
    [0.1, 0.1, 0.8],
    [0.4, 0.2, 0.4],
    [0.6, 0.3, 0.1],
    [0.4, 0.2, 0.4],
    [0.4, 0.1, 0.5],
    [0.1, 0.1, 0.8],
    [0.1, 0.8, 0.1]
])

n_classes = y_true.shape[1]  # 类别数=3

# 2. 计算每个类别的ROC和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 3. 计算Micro-average ROC
# 展平所有标签和预测概率
y_true_flat = y_true.ravel()
y_score_flat = y_score.ravel()
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_flat, y_score_flat)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# 4. 计算Macro-average ROC

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
# 取平均得到macro TPR
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# 5. 计算Weighted-average ROC
# 按每个类别的正样本数加权
class_counts = np.sum(y_true, axis=0)  # [2,5,3]
weighted_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    weighted_tpr += class_counts[i] * np.interp(all_fpr, fpr[i], tpr[i])
# 按总样本数归一化
weighted_tpr /= np.sum(class_counts)
fpr["weighted"] = all_fpr
tpr["weighted"] = weighted_tpr
roc_auc["weighted"] = auc(fpr["weighted"], tpr["weighted"])

#  6. 绘制ROC曲线
plt.figure(figsize=(10, 8), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid')

# 绘制3个类别单独的ROC曲线
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
labels = ['Class 0', 'Class 1', 'Class 2']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f'{labels[i]} (AUC = {roc_auc[i]:.3f})')

# 绘制三种平均ROC曲线
plt.plot(fpr["micro"], tpr["micro"], color='#d62728', lw=3, linestyle='--',
         label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
plt.plot(fpr["macro"], tpr["macro"], color='#9467bd', lw=3, linestyle='-.',
         label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')
plt.plot(fpr["weighted"], tpr["weighted"], color='#8c564b', lw=3, linestyle=':',
         label=f'Weighted-average (AUC = {roc_auc["weighted"]:.3f})')

# 绘制随机猜测基准线
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')

# 图表美化
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
plt.title('Multi-class ROC Curves (3 Classes + 3 Averages)', fontsize=14, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=10, framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()


plt.savefig('multi_class_roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

#  7. 输出AUC结果
print("="*50)
print("各类别及平均AUC结果：")
print("="*50)
for i in range(n_classes):
    print(f"{labels[i]} AUC: {roc_auc[i]:.4f}")
print(f"Micro-average AUC: {roc_auc['micro']:.4f}")
print(f"Macro-average AUC: {roc_auc['macro']:.4f}")
print(f"Weighted-average AUC: {roc_auc['weighted']:.4f}")
print("="*50)