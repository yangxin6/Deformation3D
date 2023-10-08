# -*- coding:utf-8 -*-
"""
coding   : utf-8
@Project : corn_organ_segmentation 
@File    : CornDataLoader.py
@IDE     : PyCharm 
@Author  : yangxin
@Date    : 2023/2/9 下午2:38 
"""

import os
import warnings
import numpy as np
from torch.utils.data import Dataset


np.random.seed(1234567)
warnings.filterwarnings('ignore')


def pc_scale(pc):
    # centroid = np.mean(pc, axis=0)
    # pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def random_rotate(xyz):
    """
    VCNN数据增强方法，随机旋转
    Args:
        xyz:

    Returns:

    """
    def rot_c3x3(rpy):
        cx, cy, cz = np.cos(rpy)  # Q: roll,pitch,yaw| trans.
        sx, sy, sz = np.sin(rpy)

        return np.array([  # R*x => y
            [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz, ],
            [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx, ],
            [-sy, cy * sx, cx * cy, ]
        ])

    ran = np.random.rand(6) * 2 - 1
    rot = ran[:3] * np.r_[5., 5., 360.] * np.pi / 180
    # trans = ran[3:] * np.r_[0., 0., 0.]
    cur_xyz = xyz.dot(rot_c3x3(rot))

    return cur_xyz


def down_sample(point_set, seg, npoints):
    """
    随机将采样
    :param point_set:
    :param seg:
    :param npoints:
    :return:
    """
    if len(seg) > npoints:
        # 随机下采样
        choice = np.random.choice(len(seg), npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]
    return point_set, seg


def fps_filter(xyzl, point_num):
    """
    FPS 下采样
    :param xyzl:
    :param point_num:
    :return:
    """
    point_cloud = xyzl[:, :3]
    filtered_points = []
    # 随机选取第一个点当做FPS下采样的起点
    point_first_index = np.random.randint(0, len(point_cloud))
    filtered_points.append(point_cloud[point_first_index])
    for i in range(int(point_num)):
        ipoint_jpoint_distance = []
        if(i == 0):     # 使用随机选取的点作为FPS的第一个点
            i_x = point_cloud[point_first_index][0]
            i_y = point_cloud[point_first_index][1]
            i_z = point_cloud[point_first_index][2]
            for j in range(len(point_cloud)):
                j_x = point_cloud[j][0]
                j_y = point_cloud[j][1]
                j_z = point_cloud[j][2]
                distance = pow((i_x-j_x), 2) + pow((i_y-j_y), 2) + pow((i_z-j_z), 2)
                ipoint_jpoint_distance.append(distance)
            distance_sort = np.argsort(ipoint_jpoint_distance)
            filtered_points.append(point_cloud[distance_sort[-1]])
            continue
        # 遍历点云中的每一个点
        for j in range(len(point_cloud)):
            j_x = point_cloud[j][0]
            j_y = point_cloud[j][1]
            j_z = point_cloud[j][2]
            distance_list = []
            # 计算遍历到的原点云中的点与已采到的点之间的距离
            for k in range(len(filtered_points)):
                point_repeat = True     # point_repeat防止比较同一个点之间的距离
                k_x = filtered_points[k][0]
                k_y = filtered_points[k][1]
                k_z = filtered_points[k][2]
                if (j_x == k_x and j_y == k_y and j_z == k_z):
                    point_repeat = False
                    break
                distance = pow((i_x-j_x), 2) + pow((i_y-j_y), 2) + pow((i_z-j_z), 2)
                distance_list.append(distance)
            if point_repeat is True:
                distance_avg = np.mean(distance_list)
                ipoint_jpoint_distance.append(distance_avg)
        distance_sort = np.argsort(ipoint_jpoint_distance)          # 对距离排序，返回索引序号
        filtered_points.append(xyzl[distance_sort[-1]])      # 将具有最大距离对应的点加入filtered_points
    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points


def fps_down_sample(xyzl, npoints):
    """
    将采样
    :param xyzl:
    :param npoints:
    :return:
    """
    if len(xyzl) > npoints:
        sample_corn_points = fps_filter(xyzl, npoints)
        return sample_corn_points
    else:
        return xyzl


class CornDataset(Dataset):
    def __init__(self,
                 root='./',
                 data_dir="data",
                 meta="meta",
                 npoints=4096,
                 split='train',
                 rotate=True,
                 ):

        self.npoints = npoints
        self.root = root
        self.meta = []
        self.split = split
        self.rotate = rotate

        self.datapath = os.path.join(self.root, data_dir)
        fns = sorted(os.listdir(self.datapath))
        # print(fns[0][0:-4])
        if split == 'trainval':
            # val_ids = [line.rstrip() for line in open(os.path.join(self.root, 'val.txt'))]
            trainval_ids = [line.rstrip() for line in open(os.path.join(self.root, meta, 'trainval.txt'))]
            fns = [fn for fn in trainval_ids]
        elif split == 'train':
            train_ids = [line.rstrip() for line in open(os.path.join(self.root, meta, 'train.txt'))]
            fns = [fn for fn in train_ids]
        elif split == 'val':
            val_ids = [line.rstrip() for line in open(os.path.join(self.root, meta, 'val.txt'))]
            fns = [fn for fn in val_ids]
        elif split == 'test':
            test_ids = [line.rstrip() for line in open(os.path.join(self.root, meta, 'test.txt'))]
            fns = [fn for fn in test_ids]
        else:
            print('Unknown split: %s. Exiting..' % (split))
            exit(-1)

        for fn in fns:
            self.meta.append(os.path.join(self.datapath, fn))

        self.cache = {}  # from index to (point_set, seg, filename) tuple
        self.cache_size = 8000

        # labelweights
        labelweights_path = os.path.join(self.root, meta, split + '-labelweights.npy')
        if os.path.exists(labelweights_path):
            self.labelweights = np.load(labelweights_path)
        else:
            self._init_labelweights(labelweights_path)

    def _init_labelweights(self, labelweights_path, N=2):
        labelweights = np.zeros(N)
        for index, item in enumerate(self.meta):
            point_set, seg, id = self._get_data(index)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, seg, id)
            tmp, _ = np.histogram(seg, range(N + 1))
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        np.save(labelweights_path, labelweights)

    def _get_data(self, index):
        fn = self.meta[index]
        id = os.path.splitext(os.path.basename(fn))[0]
        data = np.loadtxt(fn).astype(np.float32)
        seg = data[:, -1]
        point_set = data[:, 0:3]
        # 将采样到4096
        point_set, seg = down_sample(point_set, seg, self.npoints)

        # 缩放
        point_set = pc_scale(point_set)
        if self.split == 'train' or self.split == 'trainval':
            # 随机旋转
            if self.rotate:
                point_set = random_rotate(point_set)
        # 语义标签
        mask = seg == 0
        seg = np.ones_like(seg)
        seg[mask] = 0
        return point_set, seg, id

    def __getitem__(self, index):
        if index in self.cache:
            point_set, seg, id = self.cache[index]
        else:
            point_set, seg, id = self._get_data(index)
        if len(self.cache) < self.cache_size:
            self.cache[index] = (point_set, seg, id)
        return point_set, seg, id

    def __len__(self):
        return len(self.meta)


leaf_num_to_cls = {
    4: 0,
    5: 1,
    6: 2,
    7: 3,
    8: 4,
    9: 5,
    10: 6,
    11: 7,
}


class CornPartDataset(Dataset):
    """
    部件分割
    """

    def __init__(self,
                 root='./',
                 data_dir="data",
                 meta="meta",
                 npoints=4096,
                 split='train',
                 leaf_num=4
                 ):

        self.npoints = npoints
        self.root = root
        self.meta = []
        self.leaf_num = leaf_num
        self.datapath = os.path.join(self.root, data_dir)
        fns = sorted(os.listdir(self.datapath))
        # print(fns[0][0:-4])
        if split == 'trainval':
            # val_ids = [line.rstrip() for line in open(os.path.join(self.root, 'val.txt'))]
            trainval_ids = [line.rstrip() for line in open(os.path.join(self.root, meta, 'trainval.txt'))]
            fns = [fn for fn in trainval_ids]
        elif split == 'train':
            train_ids = [line.rstrip() for line in open(os.path.join(self.root, meta, 'train.txt'))]
            fns = [fn for fn in train_ids]
        elif split == 'val':
            val_ids = [line.rstrip() for line in open(os.path.join(self.root, meta, 'val.txt'))]
            fns = [fn for fn in val_ids]
        elif split == 'test':
            test_ids = [line.rstrip() for line in open(os.path.join(self.root, meta, 'test.txt'))]
            fns = [fn for fn in test_ids]
        else:
            print('Unknown split: %s. Exiting..' % (split))
            exit(-1)

        for fn in fns:
            leaf_num = int(fn.split('-')[1])
            if leaf_num != self.leaf_num:
                continue
            self.meta.append(os.path.join(self.datapath, fn))

        self.cache = {}  # from index to (point_set, seg, filename) tuple
        self.cache_size = 10000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg, id = self.cache[index]
        else:
            fn = self.meta[index]
            id = os.path.splitext(os.path.basename(fn))[0]
            leaf_num = int(id.split('-')[1])
            cls = leaf_num_to_cls[leaf_num]
            data = np.loadtxt(fn).astype(np.float32)
            point_set = data[:, 0:3]
            # point_set = pc_scale(point_set)
            label = data[:, -1].astype(np.int32)
            point_set, seg = down_sample(point_set, label, self.npoints)
            # seg = np.ones_like(label) * 100 * leaf_num
            # seg += label

        if len(self.cache) < self.cache_size:
            self.cache[index] = (point_set, cls, seg, id)
        return point_set, cls, seg, id

    def __len__(self):
        return len(self.meta)


if __name__ == '__main__':
    import torch


    data = CornDataset('/home/yangxin/datasets/3d_corn/miao_corn/deformed/20230219', meta='meta', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
    for point, label, id in DataLoader:
        if point.shape[1] < 4096:
            print(point.shape)
            print(label.shape)
