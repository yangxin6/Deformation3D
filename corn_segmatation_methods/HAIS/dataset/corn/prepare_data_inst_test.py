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
    with open(os.path.join(meta_root, 'test.txt')) as f:
        test_files = []
        for item in f.readlines():
            fn = os.path.join(data_dir, item.rstrip())
            test_files.append(fn)

    # 写入
    with open('corn_test.txt', 'w') as f:
        for item in test_files:
            key = os.path.splitext(os.path.basename(item))[0]
            f.write(f"{key}\n")


    return test_files


def generate_test_inst(fn):
    """
    组织点云数据格式，只有茎叶分割两类
    :param fn:
    :return:
    """
    xyzl = np.loadtxt(fn)
    points = xyzl[:, :3]

    rgb = np.ones_like(points).astype(np.float32)

    leaf_num = fn.split('-')[1]
    torch.save((points, rgb), 'test/' + os.path.basename(fn)[:-4] + '_' + leaf_num + '.pth')


def save_test_gt(fn):
    xyzl = np.loadtxt(fn)
    instance_label = l = xyzl[:, -1]

    mask0 = l == 0
    sem_labels = np.zeros_like(l)
    sem_labels[mask0] = 0
    mask1 = l > 0
    sem_labels[mask1] = 1

    # GT
    instance_label_new = np.zeros(instance_label.shape,
                                  dtype=np.int32)  # 0 for unannotated, xx00y: x for semantic_label, y for inst_id (1~instance_num)

    instance_num = int(instance_label.max()) + 1
    for inst_id in range(instance_num):
        instance_mask = np.where(instance_label == inst_id)[0]
        sem_id = int(sem_labels[instance_mask[0]])
        semantic_label = semantic_label_idxs[sem_id]
        instance_label_new[instance_mask] = semantic_label * 1000 + inst_id + 1

    leaf_num = fn.split('-')[1]
    np.savetxt(os.path.join('test', 'gt_' + os.path.basename(fn)[:-4] + '_' + leaf_num + '.txt'), instance_label_new, fmt='%d')


def main():
    data_root = "/home/yangxin/datasets/3d_corn/deformation/corn_txt_data_v1"
    # data_root = "/home/keys/datasets/Corn/transformed/20230310_0"
    meta_dir = os.path.join(data_root, "leaf_num_v123_meta2")
    data_dir = os.path.join(data_root, "data_pretreatment")

    mkdir_or_exist('test')

    test_files = generate_meta(meta_dir, data_dir)
    print(len(test_files))
    # fn = train_files[0]
    # generate_train_inst(fn)

    # for fn in val_files:
    #     generate_train_inst(fn)

    # 多线程
    p = mp.Pool(processes=mp.cpu_count()-2)
    p.map(generate_test_inst, test_files)
    p.map(save_test_gt, test_files)
    p.close()
    p.join()


if __name__ == "__main__":
    main()
