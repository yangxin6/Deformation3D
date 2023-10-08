# -*- coding:utf-8 -*-
"""
coding   : utf-8
@Project : corn_organ_segmentation 
@File    : post_inference.py
@IDE     : PyCharm 
@Author  : yangxin
@Date    : 2023/9/12 上午9:47 
"""

# import torch
# import time
# import random
# import os
# # import open3d as o3d

import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
from DFSP.utils import seg_items


#
# colors_txt = open('colors.txt').readlines()[0].split(';')
# label_to_color = {}
# for index, item in enumerate(colors_txt):
#     i_color = item.strip().split(' ')
#     label_to_color[index] = np.array([float(i) for i in i_color])
#
#
# def show_pcd(xyzl):
#     xyz = xyzl[:, :3]
#     l = xyzl[:, -1]
#
#     # 根据标签选择颜色
#     point_colors = [label_to_color[label] for label in l]
#
#     # 创建第一个点云对象
#     pcd1 = o3d.geometry.PointCloud()
#     pcd1.points = o3d.utility.Vector3dVector(xyz)
#     pcd1.colors = o3d.utility.Vector3dVector(point_colors)
#
#     # 创建坐标系几何对象
#     coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=np.max(xyz[:2]) / 5)
#
#     max_z = np.max(xyz[:, 2]) / 2
#
#     # 创建窗口并同时展示两个点云
#     o3d.visualization.draw_geometries([pcd1, coord_frame],
#                                       zoom=1,
#                                       front=np.array([0, 1, 0]),
#                                       lookat=np.array([0, 0, max_z]),
#                                       up=np.array([0, 0, 1]))


def assign_new_clusters(xyz, clusters_np, semantic_pred_np, res, no_clusters_idx, cluster_scores, k=5, closed_th=2,
                        npoint_th=100):
    clusters = clusters_np.copy()
    semantic_pred = semantic_pred_np.copy()
    # kdtree
    kdt = KDTree(xyz, metric='euclidean')
    query_res = kdt.query(xyz, k=k)
    # knn_radius = query_res[0][:, k - 1]
    neighbors = query_res[1]

    new_res_label = np.ones_like(res) * -1

    # 茎
    stem_idx = np.where(semantic_pred == 0)
    stem_neighbors = neighbors[stem_idx]
    stem_neighbors_items = stem_neighbors[:, 1:].flatten()
    # 茎 实例 label
    stem_c = -1
    for cc in range(clusters.shape[0]):
        stem_cc = clusters[cc][stem_idx]
        if stem_cc.sum() > len(stem_idx[0]) / 2:
            stem_c = cc

    if res[0] == -1:
        tmp_c = stem_c - 1 if stem_c == clusters.shape[0] else stem_c + 1
        res = [tmp_c for i in range(len(res))]

    new_index = 0
    for i in range(max(res) + 1):
        i_nc = np.where(res == i)
        i_nc_i = no_clusters_idx[0][i_nc]
        # # 茎
        # count = 0
        # for ii in i_nc_i:
        #     if ii in stem_neighbors_items:
        #         count += 1
        # if count > closed_th:
        #     new_res_label[i_nc] = stem_c
        #     continue
        # 大的实例
        if len(i_nc[0]) > npoint_th:
            new_res_label[i_nc] = clusters.shape[0] + new_index
            new_index += 1
            continue

    # 小实例
    single_res_idx = np.where(new_res_label == -1)
    single_xyz = xyz[no_clusters_idx[0][single_res_idx]]
    c_min_dist_list = []
    for c in clusters:
        c_p = xyz[np.where(c == 1)]
        c_dis = cdist(c_p, single_xyz, 'euclidean')
        min_dis = c_dis.min(axis=0)
        c_min_dist_list.append(min_dis)
    single_xyz_c = np.argmin(np.array(c_min_dist_list), axis=0)
    new_res_label[single_res_idx] = single_xyz_c

    # no clusters -> new_clusters
    for ii in range(min(new_res_label), max(new_res_label)+1):
        if ii < clusters.shape[0]:
            clusters[ii][no_clusters_idx[0][np.where(new_res_label == ii)]] = 1
        else:
            base_c = np.zeros(clusters.shape[1], dtype=int)
            base_c[no_clusters_idx[0][np.where(new_res_label == ii)]] = 1
            clusters = np.r_[clusters, base_c.reshape(1, -1)]
            semantic_pred[no_clusters_idx[0][np.where(new_res_label == ii)]] = 0
            cluster_scores = np.append(cluster_scores, 0.9)
        if ii == stem_c:
            semantic_pred[no_clusters_idx[0][np.where(new_res_label == ii)]] = 1

    cluster_semantic_id = np.ones(clusters.shape[0])
    cluster_semantic_id[stem_c] = 0

    return clusters, semantic_pred, cluster_semantic_id, cluster_scores


def post_process(clusters, semantic_pred, locs_float, cluster_scores, k=20, npoint_th=100):
    try:
        clusters_np = clusters.cpu().numpy()
        semantic_pred_np = semantic_pred.cpu().numpy()
        xyz_np = locs_float.cpu().numpy()

        stem_points = xyz_np[np.where(semantic_pred_np == 0)]
        no_clusters_idx = np.where(clusters_np.sum(axis=0) == 0)
        no_clusters_points = xyz_np[no_clusters_idx]

        if no_clusters_points.shape[0] > k:
            min_point = stem_points.min(axis=0)
            distance = np.linalg.norm(no_clusters_points - min_point, axis=1)
            res = seg_items(no_clusters_points, distance, k=k)
        else:
            res = np.ones(no_clusters_points.shape[0]) * -1

        new_clusters, new_semantic_pred, cluster_semantic_id, cluster_scores = assign_new_clusters(xyz_np, clusters_np,
                                                                                                   semantic_pred_np,
                                                                                                   res, no_clusters_idx,
                                                                                                   cluster_scores,
                                                                                                   npoint_th=npoint_th)
    except:
        return None, None, None, None
    # 可视化
    # xyzl = []
    # for i in range(new_clusters.shape[0]):
    #     i_points = xyz_np[np.where(new_clusters[i] == 1)]
    #     i_l = np.ones(i_points.shape[0]) * i
    #     xyzl.append(np.c_[i_points, i_l.reshape(-1, 1)])
    # xyzl_np = np.concatenate(xyzl)
    # show_pcd(xyzl_np)

    return new_clusters, new_semantic_pred, cluster_semantic_id, cluster_scores
