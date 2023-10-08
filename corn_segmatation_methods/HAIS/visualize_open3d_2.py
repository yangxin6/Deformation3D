import numpy as np
import os, glob, argparse
import torch
from operator import itemgetter
import cv2
import open3d as o3d
import glob

COLOR_DETECTRON2 = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        # 0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        # 0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        # 0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        # 0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        # 0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        # 1.000, 1.000, 1.000
    ]).astype(np.float32).reshape(-1, 3) * 255

SEMANTIC_IDXS = np.array([0, 1])
SEMANTIC_NAMES = np.array(['stem', 'leaf'])
CLASS_COLOR = {
    'unannotated': [0, 0, 0],
    'stem': [143, 223, 142],
    'leaf': [171, 198, 230],
}
SEMANTIC_IDX2NAME = {0: 'stem', 1: 'leaf'}


def get_coords_color(opt, item):
    input_file = os.path.join(item)
    assert os.path.isfile(input_file), 'File not exist - {}.'.format(input_file)
    xyz, rgb = torch.load(input_file)

    rgb = (rgb - 1) * 127.5

    if (opt.task == 'semantic_pred'):
        assert opt.data_split != 'train'
        semantic_file = os.path.join(opt.prediction_path, opt.data_split, 'semantic', os.path.basename(item)[:-4] + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(np.int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

    elif (opt.task == 'offset_semantic_pred'):
        assert opt.data_split != 'train'
        semantic_file = os.path.join(opt.prediction_path, opt.data_split, 'semantic', os.path.basename(item)[:-4] + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(np.int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

        offset_file = os.path.join(opt.prediction_path, opt.data_split, 'coords_offsets', os.path.basename(item)[:-4] + '.npy')
        assert os.path.isfile(offset_file), 'No offset result - {}.'.format(offset_file)
        offset_coords = np.load(offset_file)
        xyz = offset_coords[:, :3] + offset_coords[:, 3:]

    # same color order according to instance pointnum
    elif (opt.task == 'instance_pred'):
        assert opt.data_split != 'train'
        instance_file = os.path.join(opt.prediction_path, 'test', os.path.basename(item)[:-4] + '.txt')
        assert os.path.isfile(instance_file), 'No instance result - {}.'.format(instance_file)
        f = open(instance_file, 'r')
        masks = f.readlines()
        masks = [mask.rstrip().split() for mask in masks]
        inst_label_pred_rgb = np.zeros(rgb.shape)  # np.ones(rgb.shape) * 255 #

        ins_num = len(masks)
        ins_pointnum = np.zeros(ins_num)
        inst_label = -100 * np.ones(rgb.shape[0]).astype(int)

        for i in range(len(masks) - 1, -1, -1):
            mask_path = os.path.join(opt.prediction_path, 'test', masks[i][0])
            assert os.path.isfile(mask_path), mask_path
            if (float(masks[i][2]) < 0.09):
                continue
            mask = np.loadtxt(mask_path).astype(int)
            print('{} {}: {} pointnum: {}'.format(i, masks[i], SEMANTIC_IDX2NAME[int(masks[i][1])], mask.sum()))      
            ins_pointnum[i] = mask.sum()
            inst_label[mask == 1] = i  
        sort_idx = np.argsort(ins_pointnum)[::-1]
        for _sort_id in range(ins_num):
            inst_label_pred_rgb[inst_label == sort_idx[_sort_id]] = COLOR_DETECTRON2[_sort_id % len(COLOR_DETECTRON2)]
        rgb = inst_label_pred_rgb


    return xyz, rgb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='path to the dataset files', default='dataset/corn')
    parser.add_argument('--prediction_path', help='path to the prediction results', default='exp/corn/hais/hais_run1_corn_v1_1_elastic_500/result')
    parser.add_argument('--data_split', help='train / val / test', default='test')
    parser.add_argument('--task', help='input / semantic_gt / semantic_pred / offset_semantic_pred / instance_gt / instance_pred', default='instance_pred')
    opt = parser.parse_args()

    # test_set = glob.glob(os.path.join(opt.data_path, opt.data_split, '*pth'))

    test_set = []
    leaf_num_list = [2]
    # test_name = 'XY335-2-1-5'
    test_name = None  # XY335-2-1-3 XY335-2-1-6 XY335-2-1-5

    for item in os.listdir(os.path.join(opt.data_path, opt.data_split)):
        if item[-3:] == 'txt':
            continue
        if test_name is not None and test_name == item.split('-')[1]:
            test_set.append(os.path.join(opt.data_path, opt.data_split, item))
        else:
            item_leaf_num = int(item.split('-')[1])
            if item_leaf_num in leaf_num_list:
                test_set.append(os.path.join(opt.data_path, opt.data_split, item))

    for item in test_set:
        xyz, rgb = get_coords_color(opt, item)
        points = xyz[:, :3]
        colors = rgb / 255

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pc)
        vis.get_render_option().point_size = 1.5
        vis.run()
        vis.destroy_window()

    






