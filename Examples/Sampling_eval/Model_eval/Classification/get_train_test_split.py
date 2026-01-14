import numpy as np
from sklearn.metrics import pairwise_distances
import pandas as pd



def cvt_sampling(unlabeled_data, k, max_iters=100, tol=1e-4):
    """CVT采样算法"""
    # 1. 初始化中心点（k-means++）
    centers = unlabeled_data[np.random.choice(len(unlabeled_data), k, replace=False)]
    
    for _ in range(max_iters):
        # 2. Voronoi划分：分配样本到最近中心点
        distances = pairwise_distances(unlabeled_data, centers)
        labels = np.argmin(distances, axis=1)
        
        # 3. 更新中心点为Voronoi单元的质心
        new_centers = np.array([unlabeled_data[labels == i].mean(axis=0) for i in range(k)])
        
        # 4. 检查收敛
        if np.linalg.norm(new_centers - centers) < tol:
            break
        centers = new_centers
    
    # 返回最终中心点作为采样结果
    return centers

if __name__ == '__main__':
    np.random.seed(42)
    input_data = pd.read_csv('1700_final_norepeat.csv')
    
