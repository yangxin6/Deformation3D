# -*- coding:utf-8 -*-
"""
coding   : utf-8
@Project : corn_organ_segmentation 
@File    : utils.py
@IDE     : PyCharm 
@Author  : yangxin
@Date    : 2023/9/12 上午11:01 
"""
import QuickshiftPP
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree, BallTree


def seg_items(X, distance, k=20, ann="kdtree", beta=0.3, epsilon=0):
    n, d = X.shape
    neighbors = None

    if ann == "kdtree":
        kdt = KDTree(X, metric='euclidean')
        query_res = kdt.query(X, k=k)
        knn_radius = query_res[0][:, k - 1]
        neighbors = query_res[1]

    elif ann == "balltree":
        balltree = BallTree(X, metric='euclidean')
        query_res = balltree.query(X, k=k)
        knn_radius = query_res[0][:, k - 1]
        neighbors = query_res[1]

    memberships = np.zeros(n, dtype=np.int32)
    result = np.zeros(n, dtype=np.int32)
    neighbors = np.ndarray.astype(neighbors, dtype=np.int32)
    distance = np.ndarray.astype(distance, dtype=np.float64)
    knn_radius = 1 / distance
    X_copy = np.ndarray.astype(X, dtype=np.float64)


    QuickshiftPP.compute_mutual_knn_np(n, k, d,
                                       knn_radius,
                                       neighbors,
                                       beta, epsilon,
                                       memberships)

    # unique = np.unique(memberships)
    if (memberships == -1).all():
        result = memberships
    else:
        QuickshiftPP.cluster_remaining_np(n, k, d, X_copy, knn_radius, neighbors, memberships, result)

    return result



# def remove_outlier(xyz, clusters_np):
#     clusters = clusters_np.copy()
#
#     for cc in range(clusters.shape[0]):
#         c_i = np.where(clusters[cc] == 1)
#         i_xyz = xyz[c_i]
#         # dist = np.linalg.norm(i_xyz, axis=1)
#         kms = KMeans(n_clusters=3)
#         kms.fit(i_xyz)
#         pred_i_clusters = kms.predict(i_xyz)
#         min_i_clusters_idx = np.argmin(np.bincount(pred_i_clusters))
#         clusters[cc][np.where(pred_i_clusters == min_i_clusters_idx)] = 0
#
#     return clusters