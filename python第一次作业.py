import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

"""我的代码将每一个预测概率当作独立阈值，并且手动补充了起点 (0, 0) 和终点 (1, 1)，因此在当前无重复预测概率的数据集上，结果与 sklearn 完全相同。
如果数据集中出现多个相同的预测概率，手写代码与 sklearn 的结果就会出现差异：
我的手写代码会逐个遍历所有样本，即使预测概率相同，也会生成多个连续、重复的点，例如连续出现 (0, 0.4)、(0, 0.4) 这种无变化的共线点。
而 sklearn 会自动对相同概率阈值去重，并删除这些冗余、无变化的中间点，只保留 ROC 曲线真正的结点（拐点）。
如果手写代码保留了大量连续无变化的点（例如从 (0, 0.4) 一直到 (1, 0.4)），会让 AUC 计算时累加大量无效面积，
最终导致与 sklearn 结果出现剧烈变化、明显差异。
"""

y_true = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0])
y_score = np.array([0.90, 0.42, 0.20, 0.60, 0.50, 0.41, 0.70, 0.40, 0.65, 0.35])


#   手动实现 ROC 曲线与 AUC
def my_trapz(y_list, x_list):
    """梯形法计算面积（手动实现AUC）"""
    area = 0.0
    for i in range(1, len(x_list)):
        x0, x1 = x_list[i - 1], x_list[i]
        y0, y1 = y_list[i - 1], y_list[i]
        area += (x1 - x0) * (y0 + y1) / 2
    return area


def calculate_tpr_fpr(y_true, y_pred):
    """计算单个阈值下的TPR和FPR"""
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    P = np.sum(y_true == 1)  # 正样本总数
    N = np.sum(y_true == 0)  # 负样本总数
    TPR = TP / P if P != 0 else 0.0
    FPR = FP / N if N != 0 else 0.0
    return TPR, FPR


def my_roc_curve(y_true, y_score):
    """手动实现ROC曲线计算"""
    # 1. 按预测概率降序排序
    sorted_idx = np.argsort(-y_score)
    y_true_sorted = y_true[sorted_idx]
    y_score_sorted = y_score[sorted_idx]

    # 2. 初始化TP/FP
    TP, FP = 0, 0
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    # 3. 起点(0,0)
    tpr_list = [0.0]
    fpr_list = [0.0]

    # 4. 遍历所有样本（每个样本对应一个阈值）
    for i in range(len(y_score_sorted)):
        if y_true_sorted[i] == 1:
            TP += 1
        else:
            FP += 1
        tpr = TP / P
        fpr = FP / N
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(fpr_list), np.array(tpr_list), y_score_sorted


def my_auc_score(fpr, tpr):
    """手动计算AUC梯形法"""
    return my_trapz(tpr, fpr)


# 手动计算ROC和AUC
fpr_my, tpr_my, thresholds_my = my_roc_curve(y_true, y_score)
auc_my = my_auc_score(fpr_my, tpr_my)

# sklearn 库实现 ROC 曲线与 AUC
fpr_sk, tpr_sk, thresholds_sk = roc_curve(y_true, y_score, drop_intermediate=True)
auc_sk = roc_auc_score(y_true, y_score)

#  绘图对比
plt.rcParams["font.family"] = ["SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(8, 6))
# 手动实现ROC曲线
plt.plot(fpr_my, tpr_my, label=f'手动实现 (AUC = {auc_my:.4f})',
         linewidth=2, color='#1f77b4', marker='o', markersize=4)
# sklearn实现ROC曲线
plt.plot(fpr_sk, tpr_sk, label=f'sklearn实现 (AUC = {auc_sk:.4f})',
         linewidth=2, color='#ff7f0e', linestyle='--', marker='s', markersize=4)
# 随机猜测对角线
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机猜测 (AUC=0.5)')

plt.xlabel('假阳性率 (FPR)', fontsize=12)
plt.ylabel('真阳性率 (TPR)', fontsize=12)
plt.title('ROC曲线对比（手动实现 vs sklearn实现）', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()


print("=" * 50)
print("【手动实现结果】")
print(f"ROC-AUC值: {auc_my:.4f}")
print(f"手动FPR序列: {np.round(fpr_my, 4)}")
print(f"手动TPR序列: {np.round(tpr_my, 4)}")
print("\n【sklearn实现结果】")
print(f"ROC-AUC值: {auc_sk:.4f}")
print(f"sklearn FPR序列: {np.round(fpr_sk, 4)}")
print(f"sklearn TPR序列: {np.round(tpr_sk, 4)}")
print("=" * 50)