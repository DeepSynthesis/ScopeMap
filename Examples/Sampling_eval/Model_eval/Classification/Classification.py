import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from math import sqrt
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt

def cvt_sampling_df(data, k, not_feature_columns, max_iters=500, tol=1e-4):
    """
    CVT采样算法（支持DataFrame输入，保留非数值列）
    
    参数:
        data: DataFrame，包含特征列和非数值列（如'reactant_aldehyde', 'conv'）
        k: 需要采样的中心点数量
        feature_columns: list，参与距离计算的特征列名（数值型）
        max_iters: 最大迭代次数
        tol: 中心点变化的收敛阈值
    
    返回:
        centers: DataFrame，采样到的中心点（包含原始所有列）
        unselected_points: DataFrame，未被采样的点（原始所有列）
    """
    # 1. 提取数值特征用于CVT计算
    X = data.drop(not_feature_columns, axis=1).values
    
    # 2. 初始化中心点（从数值特征中随机选择k个）
    indices = np.random.choice(len(X), k, replace=False)
    centers_X = X[indices].copy()
    
    for _ in range(max_iters):
        # 3. Voronoi划分：计算所有点到中心点的距离
        distances = pairwise_distances(X, centers_X)
        labels = np.argmin(distances, axis=1)
        
        # 4. 更新中心点为Voronoi单元的质心
        new_centers_X = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # 5. 检查收敛
        if np.linalg.norm(new_centers_X - centers_X) < tol:
            print(f"CVT算法收敛, 迭代次数: {_}")
            break
        centers_X = new_centers_X
    
    # 6. 找到距离中心点最近的原始数据点作为最终采样点
    final_distances = pairwise_distances(X, centers_X)
    selected_indices = np.argmin(final_distances, axis=0)
    
    # 7. 构建返回的DataFrame
    centers = data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), selected_indices)
    unselected_points = data.iloc[unselected_indices].copy()
    return centers, unselected_points
def evaluate_performance(model, X, y, set_name):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]  # 获取正类概率
    acc = accuracy_score(y, y_pred)

    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='macro')

    #二分类
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_proba)
    
    print(f"\nPerformance on {set_name} set:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")
    
    return acc, recall, f1, auc

# 5. 定义Optuna目标函数
def objective(trial, X_train, y_train, X_valid, y_valid):
    # 定义搜索空间
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 6, 50),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'n_jobs': 23
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    acc, recall, f1, auc = evaluate_performance(model, X_valid, y_valid, "Validation")
    return f1

def plot_confusion_matrix(y_true, y_pred, set_name, train_size, classes=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix ({set_name} Set)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'./xgb_classify/confusion_matrix_{set_name}_binary_70_{train_size}train.png')
    plt.show()

def plot_f1_auc_vs_samples(sample_sizes, f1_scores, auc_scores):
    """
    绘制F1和AUC分数随采样数变化的折线图
    
    参数:
        sample_sizes: list，采样点数量列表
        f1_scores: list，对应的F1分数列表
        auc_scores: list，对应的AUC分数列表
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制F1和AUC折线
    plt.plot(sample_sizes, f1_scores, 'o-', label='F1 Score', color='blue', linewidth=2, markersize=8)
    plt.plot(sample_sizes, auc_scores, 's-', label='AUC Score', color='red', linewidth=2, markersize=8)
    
    # 设置图表属性
    plt.xlabel('Sample Size', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('F1 and AUC Scores vs Sample Size', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴范围
    plt.ylim(0, 1)
    plt.xlim(min(sample_sizes) - 5, max(sample_sizes) + 5)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('./xgb_classify/f1_auc_vs_samples.png', dpi=300, bbox_inches='tight')
    plt.savefig('./xgb_classify/f1_auc_vs_samples.svg', bbox_inches='tight')
    plt.show()

def grade_xgb_model_test(train_data, unselected_points, test_percentage, not_feature_columns, y_column):
    X_train = train_data.drop(not_feature_columns, axis=1).values
    y_train = pd.DataFrame([float(y.split('%')[0])/100.0 for y in train_data[y_column]], index=None)
    y_train = y_train[0].apply(lambda x: 1 if x > 0.7 else 0)

    y_unselected = pd.DataFrame([float(y.split('%')[0])/100.0 for y in unselected_points[y_column]], index=None)
    y_unselected = y_unselected[0].apply(lambda x: 1 if x > 0.7 else 0)

    X_valid, X_test, y_valid, y_test = train_test_split(unselected_points.drop(not_feature_columns, axis=1).values, y_unselected, test_size=test_percentage, random_state=42)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(partial(objective, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid), n_trials=800)

    print("\n=== Optuna Optimization Results ===")
    print("Best trial:")
    trial = study.best_trial
    print(f"  R2: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # 7. 使用最佳参数训练最终模型
    best_params = trial.params
    best_params.update({'random_state': 42, 'n_jobs': -1})
    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)

    # 8. 评估模型性能
    print("\n=== Final Model Evaluation ===")
    _ = evaluate_performance(best_model, X_train, y_train, "Training")
    _ = evaluate_performance(best_model, X_valid, y_valid, "Validation")
    acc_test, recall_test, f1_test, auc_test = evaluate_performance(best_model, X_test, y_test, "Test")

    # 9. 特征重要性分析
    print("\n=== Feature Importance ===")
    feature_importances = pd.DataFrame({
        'Feature': train_data.drop(not_feature_columns, axis=1).columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    feature_importances.to_csv(f'./xgb_classify/feature_importance_binary_70_{len(X_train)}train.csv')

    print(feature_importances.head(10))
    classes = y_unselected.unique() if len(y_unselected.unique()) <= 10 else None  # 避免过多类别导致图像混乱
    plot_confusion_matrix(y_train, best_model.predict(X_train), "Training", len(X_train), classes)
    plot_confusion_matrix(y_valid, best_model.predict(X_valid), "Validation", len(X_train), classes)
    plot_confusion_matrix(y_test, best_model.predict(X_test), "Test", len(X_train), classes)
    
    # 返回测试集的F1和AUC分数
    return f1_test, auc_test

# 示例用法
if __name__ == "__main__":
    np.random.seed(42)
    data = pd.read_csv('1700_final_norepeat.csv')
    
    # 指定参与距离计算的特征列
    not_feature_cols = ['reactant_aldehyde', 'conv']
    
    # 存储结果用于绘图
    sample_sizes = []
    f1_scores = []
    auc_scores = []
    
    # 执行CVT采样
    for i in range(5):
        k = (i + 1) * 20 
        sample_sizes.append(k)
        
        sampled_points, unsampled_points = cvt_sampling_df(
            data=data,
            k=k,
            not_feature_columns=not_feature_cols
        )
        sampled_points.to_csv(f'./xgb_classify/sampled_points_binary_70_{k}k.csv', index=False)
        f1_test, auc_test = grade_xgb_model_test(sampled_points, unsampled_points, 0.8, not_feature_cols, 'conv')
        
        f1_scores.append(f1_test)
        auc_scores.append(auc_test)
    
    # 绘制F1和AUC随采样数变化的折线图
    plot_f1_auc_vs_samples(sample_sizes, f1_scores, auc_scores)
    
    # 保存结果到CSV
    results_df = pd.DataFrame({
        'Sample_Size': sample_sizes,
        'F1_Score': f1_scores,
        'AUC_Score': auc_scores
    })
    results_df.to_csv('./xgb_classify/f1_auc_results.csv', index=False)
    print("\n=== F1 and AUC Results ===")
    print(results_df)

