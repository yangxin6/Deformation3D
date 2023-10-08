# -*- coding:utf-8 -*-
"""
coding   : utf-8
@Project : corn_organ_segmentation 
@File    : prepare_data_inst.py
@IDE     : PyCharm 
@Author  : yangxin
@Date    : 2023/3/13 下午7:16 
"""
import os
import multiprocessing as mp
import numpy as np
import torch

semantic_label_idxs = [0, 1]
semantic_label_names = ['stem', 'leaf']

data_name = "leaf_num_v1_100"
meta_name = "meta_100"
data_root = "/home/yangxin/datasets/3d_corn/deformation/deformed/data_leaf_num_v1_20230402"

def mkdir_or_exist(dir_name, mode=0o777):
    """
    递归创建文件夹
    Args:
        dir_name:
        mode:

    Returns:

    """
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def generate_meta(meta_root, data_dir):
    """
    生成训练 meta
    :param meta_root:
    :param data_dir:
    :return:
    """
    with open(os.path.join(meta_root, 'train.txt')) as f:
        train_files = []
        for item in f.readlines():
            fn = os.path.join(data_dir, item.rstrip())
            train_files.append(fn)

    with open(os.path.join(meta_root, 'val.txt')) as f:
        val_files = []
        for item in f.readlines():
            fn = os.path.join(data_dir, item.rstrip())
            val_files.append(fn)

    # 写入
    with open(data_name + '/corn_train.txt', 'w') as f:
        for item in train_files:
            key = os.path.splitext(os.path.basename(item))[0]
            f.write(f"{key}\n")

    with open(data_name + '/corn_val.txt', 'w') as f:
        for item in val_files:
            key = os.path.splitext(os.path.basename(item))[0]
            f.write(f"{key}\n")

    return train_files, val_files


def generate_train_inst(fn):
    """
    组织点云数据格式，只有茎叶分割两类
    :param fn:
    :param data_dir:
    :param target_dir:
    :return:
    """
    xyzl = np.loadtxt(fn)
    points = xyzl[:, :3]
    l = xyzl[:, -1]

    mask0 = l == 0
    sem_labels = np.zeros_like(l)
    sem_labels[mask0] = 0
    mask1 = l > 0
    sem_labels[mask1] = 1

    instance_labels = l

    rgb = np.ones_like(points).astype(np.float32)

    leaf_num = fn.split('-')[1]
    torch.save((points, rgb, sem_labels, instance_labels), data_name + '/train/' + os.path.basename(fn)[:-4] + '_' + leaf_num + '.pth')


def generate_val_inst(fn):
    """
    组织点云数据格式，只有茎叶分割两类
    :param fn:
    :param data_dir:
    :param target_dir:
    :return:
    """
    xyzl = np.loadtxt(fn)
    points = xyzl[:, :3]
    l = xyzl[:, -1]

    mask0 = l == 0
    sem_labels = np.zeros_like(l)
    sem_labels[mask0] = 0
    mask1 = l > 0
    sem_labels[mask1] = 1

    instance_labels = l

    rgb = np.ones_like(points).astype(np.float32)

    leaf_num = fn.split('-')[1]
    torch.save((points, rgb, sem_labels, instance_labels),
               data_name + '/val/' + os.path.basename(fn)[:-4] + '_' + leaf_num + '.pth')

def main():

    # data_root = "/home/keys/datasets/Corn/transformed/20230310_0"
    meta_dir = os.path.join(data_root, meta_name)
    data_dir = os.path.join(data_root, "data")

    mkdir_or_exist(data_name + '/train')
    mkdir_or_exist(data_name + '/val')

    train_files, val_files = generate_meta(meta_dir, data_dir)

    # fn = train_files[0]
    # generate_train_inst(fn)

    # for fn in val_files:
    #     generate_train_inst(fn)

    # 多线程
    p = mp.Pool(processes=mp.cpu_count()-2)
    p.map(generate_train_inst, train_files)
    p.map(generate_val_inst, val_files)
    # p.map(save_val_gt, val_files)
    p.close()
    p.join()


if __name__ == "__main__":
    main()
