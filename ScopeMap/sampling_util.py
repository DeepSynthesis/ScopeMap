import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

def compute_cluster_centroid(cluster_points, metric='euclidean'):
    """
    Calculate cluster centroid based on distance metric
    
    Parameters:
        cluster_points: numpy.ndarray, data points in the cluster
        metric: str, distance metric method
    
    Returns:
        numpy.ndarray, cluster centroid
    """
    if len(cluster_points) == 0:
        raise ValueError("Cluster point set cannot be empty")
    
    if metric == 'manhattan':
        return np.median(cluster_points, axis=0)
    else:
        return np.mean(cluster_points, axis=0)

def compute_pairwise_distances(X, Y=None, metric='euclidean'):
    """
    Compute pairwise distances between data points, supporting multiple distance metrics
    
    Parameters:
        X: numpy.ndarray, data point matrix
        Y: numpy.ndarray, optional, second set of data points. If None, compute distances within X
        metric: str, distance metric method, supports 'euclidean', 'manhattan', 'cosine'
    
    Returns:
        numpy.ndarray, distance matrix
    """
    if metric in ['euclidean', 'manhattan']:
        return pairwise_distances(X, Y, metric=metric)
    elif metric == 'cosine':
        return pairwise_distances(X, Y, metric='cosine')
    else:
        raise ValueError(f"Unsupported distance metric: {metric}. Supported metrics: 'euclidean', 'manhattan', 'cosine'")


def cvt_sampling_df(data, k, not_feature_columns, max_iters=500, tol=1e-4):
    """
    CVT sampling algorithm (supports DataFrame input, preserves non-numeric columns)
    
    Parameters:
        data: DataFrame, contains feature columns and non-numeric columns (e.g., 'reactant_aldehyde', 'conv')
        k: number of center points to sample
        not_feature_columns: list, column names not participating in distance calculation (numeric type)
        max_iters: maximum number of iterations
        tol: convergence threshold for center point changes
    
    Returns:
        centers: DataFrame, sampled center points (containing all original columns)
        unselected_points: DataFrame, points not sampled (all original columns)
    """
    X = data.drop(not_feature_columns, axis=1).values
    
    indices = np.random.choice(len(X), k, replace=False)
    centers_X = X[indices].copy()
    
    for _ in range(max_iters):
        if np.isnan(centers_X).any():
            centers_X[np.isnan(centers_X)] = np.mean(centers_X[~np.isnan(centers_X)])
        distances = pairwise_distances(X, centers_X)

        labels = np.argmin(distances, axis=1)
        
        new_centers_X = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        if np.linalg.norm(new_centers_X - centers_X) < tol:
            print(f"CVT algorithm converged, iterations: {_}")
            break
        centers_X = new_centers_X
    
    if np.isnan(centers_X).any():
        centers_X[np.isnan(centers_X)] = np.mean(centers_X[~np.isnan(centers_X)])
    final_distances = pairwise_distances(X, centers_X)
    selected_indices = np.argmin(final_distances, axis=0)
    
    centers = data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), selected_indices)
    unselected_points = data.iloc[unselected_indices].copy()
    return centers, unselected_points

def classify_by_centers(data, centers, not_feature_columns):
    """
    Get the classification for a set of data
    
    Parameters:
        data: DataFrame, contains feature columns and non-numeric columns (e.g., 'reactant_aldehyde', 'conv')
        centers: DataFrame, contains feature columns and non-numeric columns (e.g., 'reactant_aldehyde', 'conv')
        not_feature_columns: list, column names of non-descriptive columns
    
    Returns:
        output: DataFrame, with added "labels" classification column (containing all original columns)
        Note: categories correspond to centers indices
    """
    raw_data = data.copy()
    for not_feature_column in not_feature_columns:
        if not_feature_column in data.columns:
            data = data.drop(not_feature_column, axis=1)
        if not_feature_column in centers.columns:
            centers = centers.drop(not_feature_column, axis=1)
    data = data.values
    centers = centers.values
    distances = np.sqrt(((data[:, np.newaxis, :] - centers) ** 2).sum(axis=2))
    
    labels = np.argmin(distances, axis=1).astype(int)
    if 'labels' in raw_data.columns:
        raw_data = raw_data.drop('labels', axis=1)
    output_data = pd.concat([raw_data, pd.DataFrame(labels, columns=['labels'])], axis=1)

    
    return output_data


def get_sampling(task_itr_id, drop_classes, not_feature_cols, k, max_iters=500, tol=1e-4):
    '''
    Perform one sampling and classification based on sampling.
    
    Parameters:
        task_itr_id: int, iteration number (starting from 1)
        drop_classes: list, classes to exclude
        not_feature_cols: list, column names of non-feature columns
        k: int, number of sampling center points
        max_iters: int, maximum number of iterations
        tol: float, convergence threshold for center point changes
    
    Returns:
        sampled_points: DataFrame, sampled center points (containing all original columns, i.e., label is from previous iteration)
        labeled_points: DataFrame, data and categories after sampling classification (containing all original columns, categories are sampled_points indices)
    '''
    data = pd.read_csv(f'./itr/labeled_points_itr{task_itr_id-1}.csv')
    
    if drop_classes != []:
        for drop_class in drop_classes:
            data = data[data['labels'] != drop_class].reset_index(drop=True)
    print(data)
    sampled_points, unsampled_points = cvt_sampling_df(
        data=data,
        k=k,
        not_feature_columns=not_feature_cols,
        max_iters=max_iters,
        tol=tol
    )
    sampled_points.to_csv(f'./itr/sampled_points_itr{task_itr_id}.csv', index=False)
    labeled_points = classify_by_centers(data, sampled_points, not_feature_cols)
    labeled_points.to_csv(f'./itr/labeled_points_itr{task_itr_id}.csv', index=False)
    return sampled_points, labeled_points

def get_sampling_weighted(data, task_itr_id, not_feature_cols, k, max_iters=500, tol=1e-4):
    '''
    Perform one sampling and classification based on sampling.
    
    Parameters:
        task_itr_id: int, iteration number (starting from 1)
        not_feature_cols: list, column names of non-feature columns
        k: int, number of sampling center points
        max_iters: int, maximum number of iterations
        tol: float, convergence threshold for center point changes
    
    Returns:
        sampled_points: DataFrame, sampled center points (containing all original columns, i.e., label is from previous iteration)
        labeled_points: DataFrame, data and categories after sampling classification (containing all original columns, categories are sampled_points indices)
    '''
    sampled_data = data[data['ScreenLabel']!='BASE']
    print(data)
    sampled_points, unsampled_points = weighted_itr_cvt_sampling_df_norepeat(
        data=data,
        k=k,
        not_feature_columns=not_feature_cols,
        max_iters=max_iters,
        tol=tol,
        sampled_data=sampled_data,
    )
    labeled_points = classify_by_centers(data, sampled_points, not_feature_cols)
    return sampled_points, labeled_points
def weighted_itr_cvt_sampling_df_norepeat(data, k, not_feature_columns, max_iters=2000, tol=1e-5, 
                                        sampled_data=None, repulsion_strength=0.0, cvt_weight=1.0,
                                        learning_rate=0.01, adaptive_lr=True, distance_metric='euclidean',
                                        cvt_init_iters=100):
    """
    Weighted iterative CVT sampling algorithm: combines CVT energy minimization with inverse square repulsion force from already sampled points
    Improved version: first use analytical CVT method for good initial guess, then perform gradient descent optimization
    
    Parameters:
        data: DataFrame, contains feature columns and non-numeric columns
        k: number of center points to sample
        not_feature_columns: list, column names not participating in distance calculation
        max_iters: maximum number of iterations
        tol: convergence threshold
        sampled_data: DataFrame, already sampled data points (immovable, generate repulsion force). Only calculate repulsion force for points with ScreenLabel='Excluded_Sampled'
        repulsion_strength: float, repulsion force strength coefficient
        cvt_weight: float, CVT energy weight
        learning_rate: float, gradient descent learning rate
        adaptive_lr: bool, whether to use adaptive learning rate
        distance_metric: str, distance metric method, supports 'euclidean', 'manhattan', 'cosine'
        cvt_init_iters: int, number of iterations for initial CVT analytical optimization
    
    Returns:
        centers: DataFrame, sampled center points
        unselected_points: DataFrame, unsampled points
    """
    
    def compute_distances_metric(points, centers, metric):
        """计算指定度量的距离"""
        return compute_pairwise_distances(points, centers, metric)
    
    def compute_cvt_energy_gradients(movable_centers, fixed_centers, data_points, metric='euclidean'):
        """
        计算CVT能量和梯度，包含可移动中心点和固定中心点
        
        参数:
            movable_centers: 可移动的中心点（本次采样点），参与梯度优化
            fixed_centers: 固定的中心点（已采样点），不参与梯度优化
            data_points: 所有数据点
            metric: 距离度量方法
        
        返回:
            energy: CVT能量
            gradients: 仅针对可移动中心点的梯度
        """
        if fixed_centers is not None and len(fixed_centers) > 0:
            all_centers = np.vstack([movable_centers, fixed_centers])
        else:
            all_centers = movable_centers
        
        distances = compute_distances_metric(data_points, all_centers, metric)
        labels = np.argmin(distances, axis=1)
        
        energy = 0
        gradients = np.zeros_like(movable_centers)
        
        for i in range(len(all_centers)):
            cluster_points = data_points[labels == i]
            if len(cluster_points) > 0:
                diff = cluster_points - all_centers[i]
                if metric == 'euclidean':
                    energy += np.sum(diff**2)
                elif metric == 'manhattan':
                    energy += np.sum(np.abs(diff))
                elif metric == 'cosine':
                    energy += np.sum(diff**2)
        
        for i in range(len(movable_centers)):
            cluster_points = data_points[labels == i]
            if len(cluster_points) > 0:
                diff = cluster_points - movable_centers[i]
                if metric == 'euclidean':
                    gradients[i] = -2 * np.sum(diff, axis=0)
                elif metric == 'manhattan':
                    gradients[i] = -np.mean(np.sign(diff), axis=0)
                elif metric == 'cosine':
                    gradients[i] = -2 * np.sum(diff, axis=0)
        
        return energy, gradients
    
    def compute_repulsion_energy_gradients(centers, repulsion_points, strength, metric='euclidean'):
        """计算排斥能量和梯度"""
        energy = 0
        gradients = np.zeros_like(centers)
        
        for i, center in enumerate(centers):
            for repulsion_point in repulsion_points:
                diff = center - repulsion_point
                
                if metric == 'euclidean':
                    distance_sq = np.sum(diff**2) + 1e-10
                    energy += strength / distance_sq
                    gradients[i] += -2 * strength * diff / (distance_sq**2)
                elif metric == 'manhattan':
                    manhattan_dist = np.sum(np.abs(diff)) + 1e-10
                    energy += strength / (manhattan_dist**2)
                    gradients[i] += -2 * strength * np.sign(diff) / (manhattan_dist**3)
                elif metric == 'cosine':
                    distance_sq = np.sum(diff**2) + 1e-10
                    energy += strength / distance_sq
                    gradients[i] += -2 * strength * diff / (distance_sq**2)
        
        return energy, gradients
    
    def compute_total_energy_and_gradients(movable_centers, fixed_centers, data_points, repulsion_points, 
                                         repulsion_strength, cvt_weight, metric):
        """计算总能量和梯度"""
        cvt_energy, cvt_gradients = compute_cvt_energy_gradients(movable_centers, fixed_centers, data_points, metric)
        
        if repulsion_points is not None and len(repulsion_points) > 0:
            repulsion_energy, repulsion_gradients = compute_repulsion_energy_gradients(
                movable_centers, repulsion_points, repulsion_strength, metric)
        else:
            repulsion_energy = 0
            repulsion_gradients = np.zeros_like(movable_centers)
        
        total_energy = cvt_weight * cvt_energy + repulsion_energy
        total_gradients = cvt_weight * cvt_gradients + repulsion_gradients
        
        return total_energy, total_gradients
    
    all_data_features = data.drop(not_feature_columns, axis=1).values
    print("Calculating maximum distance between two points in data...")
    
    distance_matrix = compute_pairwise_distances(all_data_features, all_data_features, distance_metric)
    max_distance = np.max(distance_matrix)
    
    print(f"Maximum distance in data: {max_distance:.6f}")
    
    d = all_data_features.shape[1]
    expected_sq_distance = (max_distance**2) * d / 12
    print(f"Expected square distance between sampling points under uniform sampling: {expected_sq_distance:.6f}")
    
    n_points = len(all_data_features)
    base_repulsion_coefficient = expected_sq_distance**2 * n_points
    
    final_repulsion_strength = base_repulsion_coefficient * (10 ** repulsion_strength)
    
    print(f"Base repulsion coefficient: {base_repulsion_coefficient:.6e}")
    print(f"Final repulsion strength (base * 10^{repulsion_strength}): {final_repulsion_strength:.6e}")
    
    if sampled_data is not None:
        available_data = data[data['ScreenLabel'] == 'BASE'].copy()
        
        all_sampled_features = sampled_data.drop(not_feature_columns, axis=1).values
        print(f"Found {len(all_sampled_features)} already sampled points as CVT fixed center points")
        
        if 'ScreenLabel' in sampled_data.columns:
            excluded_sampled_data = sampled_data[sampled_data['ScreenLabel'] == 'Excluded_Sampled']
            if len(excluded_sampled_data) > 0:
                repulsion_features = excluded_sampled_data.drop(not_feature_columns, axis=1).values
                print(f"Using {len(excluded_sampled_data)} points with ScreenLabel='Excluded_Sampled' for repulsion force calculation")
            else:
                repulsion_features = None
                print("No points with ScreenLabel='Excluded_Sampled' found, no repulsion force calculated")
        else:
            repulsion_features = None
            print("ScreenLabel column not found, no repulsion force calculated")
    
        all_data_features = data.drop(not_feature_columns, axis=1).values
        print(f"CVT calculation will use all {len(all_data_features)} data points")
    else:
        available_data = data.copy()
        all_sampled_features = None
        repulsion_features = None
        all_data_features = data.drop(not_feature_columns, axis=1).values
        print("No sampled data available, performing standard CVT sampling")
    
    X = available_data.drop(not_feature_columns, axis=1).values
    
    if len(X) < k:
        raise ValueError(f"Available data points ({len(X)}) is less than required sampling count ({k})")
    
    print(f"Stage 1: Initializing center points using analytical CVT method ({cvt_init_iters} iterations)")
    
    indices = np.random.choice(len(X), k, replace=False)
    centers = X[indices].copy().astype(float)
    
    cvt_data_points = X
    
    for cvt_iter in range(cvt_init_iters):
        if np.isnan(centers).any():
            centers[np.isnan(centers)] = np.mean(centers[~np.isnan(centers)])
        
        if all_sampled_features is not None and len(all_sampled_features) > 0:
            all_centers_for_voronoi = np.vstack([centers, all_sampled_features])
        else:
            all_centers_for_voronoi = centers
        
        distances = compute_distances_metric(cvt_data_points, all_centers_for_voronoi, distance_metric)
        labels = np.argmin(distances, axis=1)
        
        new_centers = np.zeros_like(centers)
        for i in range(k):
            cluster_points = cvt_data_points[labels == i]
            if len(cluster_points) > 0:
                new_centers[i] = compute_cluster_centroid(cluster_points, distance_metric)
            else:
                new_centers[i] = centers[i]
        
        if np.linalg.norm(new_centers - centers) < tol:
            print(f"Analytical CVT converged, iterations: {cvt_iter}")
            break
        centers = new_centers
    
    print(f"Stage 2: Weighted energy optimization based on analytical CVT initial guess (max {max_iters} iterations)")
    
    prev_energy = float('inf')
    lr = learning_rate
    
    print(f"Starting weighted energy optimization, repulsion strength: {final_repulsion_strength:.6e}, CVT weight: {cvt_weight}")
    
    for iteration in range(max_iters):
        total_energy, gradients = compute_total_energy_and_gradients(
            centers, all_sampled_features, X, repulsion_features, final_repulsion_strength, cvt_weight, distance_metric)
        
        centers_new = centers - lr * gradients
        
        if adaptive_lr:
            new_energy, _ = compute_total_energy_and_gradients(
                centers_new, all_sampled_features, X, repulsion_features, final_repulsion_strength, cvt_weight, distance_metric)
            
            if new_energy > total_energy:
                lr *= 0.8
                if lr < learning_rate * 1e-4:
                    print(f"Learning rate too small, early stopping. Iterations: {iteration}")
                    break
                continue
            elif abs(new_energy - prev_energy) < tol * abs(prev_energy) if prev_energy != 0 else tol:
                print(f"Weighted energy optimization converged, iterations: {iteration}, final energy: {new_energy:.6f}")
                break
            else:
                lr = min(lr * 1.05, learning_rate)
        
        centers = centers_new
        prev_energy = total_energy
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Total energy = {total_energy:.6f}, Learning rate = {lr:.6f}")
    
    final_distances = compute_distances_metric(X, centers, distance_metric)
    
    selected_indices = []
    used_indices = set()
    
    for center_idx in range(k):
        distances_to_center = final_distances[:, center_idx]
        candidate_indices = np.argsort(distances_to_center)
        
        for candidate_idx in candidate_indices:
            if candidate_idx not in used_indices:
                selected_indices.append(candidate_idx)
                used_indices.add(candidate_idx)
                break
        else:
            raise ValueError(f"Unable to find enough non-duplicate sampling points. Need {k} points, but can only find {len(selected_indices)} non-duplicate points")
    
    selected_indices = np.array(selected_indices)
    
    result_centers = available_data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), result_centers.index)
    unselected_points = data.iloc[unselected_indices].copy()
    
    print(f"Weighted CVT sampling completed, sampled {len(result_centers)} points")
    return result_centers, unselected_points