import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
import open3d as o3d
from DFSP.utils import seg_items

import matplotlib.pyplot as plt


def visualize_dbscan(pcd, eps=0.02, min_points=10):
    """
    Visualize the result of DBSCAN clustering.

    :param pcd: Open3D point cloud object or numpy array of shape (N, 3).
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_points: The number of samples in a neighborhood for a point to be considered as a core point.
    """
    if isinstance(pcd, np.ndarray):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    else:
        pcd_o3d = pcd

    labels = np.array(pcd_o3d.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # Setting noisy points to black
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pcd_o3d])

def apply_dbscan_to_clusters(xyz, eps=0.02, min_points=10):
    """
    应用DBSCAN来识别可能包含两个叶子的实例。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    # 若识别出的簇数量大于1，则认为可能包含两个叶子
    return len(set(labels)) - (1 if -1 in labels else 0) > 1

def post_process(clusters, semantic_pred, locs_float, cluster_scores, k=20, npoint_th=100, eps=0.05, min_points=100):
    clusters_np = clusters.cpu().numpy()
    semantic_pred_np = semantic_pred.cpu().numpy()
    xyz_np = locs_float.cpu().numpy()

    # sematic 0
    semantic_filter = semantic_pred_np > 0

    stem_idx = np.where(semantic_pred_np == 0)
    # 改进的stem_c计算方法
    stem_c = -1
    max_stem_ratio = -1
    stem_points_in_cluster_list = []
    # 统计出其他的茎，然后将作为未聚类的点
    for cc in range(clusters_np.shape[0]):
        cluster_points_idx = np.where(clusters_np[cc] == 1)[0]
        stem_points_in_cluster = np.intersect1d(cluster_points_idx, stem_idx[0])
        stem_points_in_cluster_list.append(stem_points_in_cluster)
        stem_ratio = len(stem_points_in_cluster) / len(cluster_points_idx) if len(cluster_points_idx) > 0 else 0

        if stem_ratio > max_stem_ratio:
            max_stem_ratio = stem_ratio
            stem_c = cc

    # for i in range(len(stem_points_in_cluster_list)):
    #     if i == stem_c:
    #         continue
    #     if len(stem_points_in_cluster_list[i]) == 0:
    #         continue
    #     stem_points_in_cluster = stem_points_in_cluster_list[i]
    #     clusters_np[stem_points_in_cluster] = 0

    # stem_c_mask = clusters_np[stem_c]
    # clusters_np = np.delete(clusters_np, stem_c)

    # 识别并处理可能包含两个叶子的实例
    new_zero_clusters = []
    for i in range(clusters_np.shape[0]):
        if i == stem_c:
            continue
        i_idx = np.where(clusters_np[i] == 1)
        cluster_points = xyz_np[i_idx]
        if len(cluster_points) == 0:
            continue
        # np.savetxt('111.txt', cluster_points)
        # visualize_dbscan(cluster_points, eps, min_points)
        if apply_dbscan_to_clusters(cluster_points, eps=eps, min_points=min_points):
            # 将这些点标记为未识别状态

            # clusters_np[0, i_idx] = 1  # 清除聚类标记
            new_zero_clusters.append(i)
            # semantic_pred_np[np.where(clusters_np[i] == 1)] = -1  # 在semantic_pred_np中标记为-1

    for idx in new_zero_clusters:
        clusters_np = np.delete(clusters_np, idx, axis=0)
    # 未识别点云的索引
    no_clusters_idx = np.where(clusters_np.sum(axis=0) == 0)
    if len(no_clusters_idx[0]) == 0:
        return None, None, None, None
    no_clusters_points = xyz_np[no_clusters_idx]
    # 以下是未修改的DFSP处理逻辑
    # try:
    if no_clusters_points.shape[0] > k:
        min_point = xyz_np[np.where(semantic_pred_np == 0)].min(axis=0)
        distance = np.linalg.norm(xyz_np[no_clusters_idx] - min_point, axis=1)
        res = seg_items(xyz_np[no_clusters_idx], distance, k=k)
    else:
        res = np.ones(no_clusters_points.shape[0]) * -1

    new_clusters, new_semantic_pred, cluster_semantic_id, cluster_scores = assign_new_clusters(xyz_np, clusters_np, semantic_pred_np, res, no_clusters_idx, cluster_scores, npoint_th=npoint_th)
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     return None, None, None, None
    return new_clusters, new_semantic_pred, cluster_semantic_id, cluster_scores


def assign_new_clusters(xyz, clusters_np, semantic_pred_np, res, no_clusters_idx, cluster_scores, k=5, npoint_th=100):
    res = res.astype(int)
    clusters = clusters_np.copy()
    semantic_pred = semantic_pred_np.copy()
    kdt = KDTree(xyz, metric='euclidean')
    _, neighbors = kdt.query(xyz, k=k)

    new_res_label = np.ones_like(res) * -1


    # if res[0] == -1:
    #     tmp_c = stem_c - 1 if stem_c == clusters.shape[0] else stem_c + 1
    #     res = [tmp_c for _ in range(len(res))]

    new_index = 0

    for i in range(max(res) + 1):
        i_nc = np.where(res == i)
        # i_nc_i = no_clusters_idx[0][i_nc]

        if len(i_nc[0]) > npoint_th:
            new_res_label[i_nc] = clusters.shape[0] + new_index
            new_index += 1

    # 改进的stem_c计算方法
    stem_idx = np.where(semantic_pred == 0)
    stem_c0 = -1
    max_stem_ratio = -1
    for cc in range(clusters.shape[0]):
        cluster_points_idx = np.where(clusters[cc] == 1)[0]
        stem_points_in_cluster = np.intersect1d(cluster_points_idx, stem_idx[0])
        stem_ratio = len(stem_points_in_cluster) / len(cluster_points_idx) if len(cluster_points_idx) > 0 else 0

        if stem_ratio > max_stem_ratio:
            max_stem_ratio = stem_ratio
            stem_c0 = cc

    single_res_idx = np.where(new_res_label == -1)
    if len(single_res_idx[0]) != 0:
        single_xyz = xyz[no_clusters_idx[0][single_res_idx]]
        c_min_dist_list = []
        for c in clusters:
            c_p = xyz[np.where(c == 1)]
            # if len(c_p) == 0:
            #     continue
            c_dis = cdist(c_p, single_xyz, 'euclidean')
            min_dis = c_dis.min(axis=0)
            c_min_dist_list.append(min_dis)
        # single_xyz_c = np.argmin(np.min(np.array(c_min_dist_list), axis=0))
        # single_xyz_c = np.argmin(np.array(c_min_dist_list), axis=1)
        single_xyz_c = np.argmin(np.array(c_min_dist_list), axis=0)

        # if stem_c0 in single_xyz_c:
        #     new_res_label[single_res_idx] = stem_c0
        # else:
        #     new_res_label[single_res_idx] = np.argmax(np.bincount(single_xyz_c))
        new_res_label[single_res_idx] = single_xyz_c

    new_res_labels = np.unique(new_res_label).astype(int)
    for ii in new_res_labels:
        if ii < clusters.shape[0]:
            clusters[ii][no_clusters_idx[0][np.where(new_res_label == ii)]] = 1
        else:
            base_c = np.zeros(clusters.shape[1], dtype=int)
            base_c[no_clusters_idx[0][np.where(new_res_label == ii)]] = 1
            clusters = np.r_[clusters, base_c.reshape(1, -1)]
            # semantic_pred[no_clusters_idx[0][np.where(new_res_label == ii)]] = 1
            cluster_scores = np.append(cluster_scores, 0.9)
        # semantic_pred[no_clusters_idx[0][np.where(new_res_label == ii)]] = 0 if ii == stem_c else 1

    stem_idx = np.where(semantic_pred == 0)

    # 改进的stem_c计算方法
    stem_c = -1
    max_stem_ratio = -1
    for cc in range(clusters.shape[0]):
        cluster_points_idx = np.where(clusters[cc] == 1)[0]
        stem_points_in_cluster = np.intersect1d(cluster_points_idx, stem_idx[0])
        stem_ratio = len(stem_points_in_cluster) / len(cluster_points_idx) if len(cluster_points_idx) > 0 else 0

        if stem_ratio > max_stem_ratio:
            max_stem_ratio = stem_ratio
            stem_c = cc

    cluster_semantic_id = np.ones(clusters.shape[0])
    cluster_semantic_id[stem_c] = 0

    return clusters, semantic_pred, cluster_semantic_id, cluster_scores
