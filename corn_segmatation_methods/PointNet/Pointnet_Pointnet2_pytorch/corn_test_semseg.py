"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import time

from data_utils.CornDataLoader import CornDataset
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score


def compute_iou(y_true, y_pred, num_classes=2):
    # 获取混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    # 计算 IoU
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection

    iou = intersection / union.astype(np.float32)
    return iou


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['stem', 'leaf']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--meta', type=str, default="meta", help='meta dir')
    parser.add_argument('--num_votes', type=int, default=5,
                        help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()


def mkdir_or_exist(dir_name, mode=0o777):
    """
    递归创建文件夹
    :param dir_name:
    :param mode:
    :return:
    """
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    # visual_dir = experiment_dir + '/visual/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()) + '/'
    visual_dir = experiment_dir + '/visual/'
    mkdir_or_exist(visual_dir)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 2
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point
    VISUAL = args.visual
    NUM_VOTES = args.num_votes
    META = args.meta

    root = '/home/yangxin/datasets/3d_corn/deformation2/corn_txt_data_v1/'
    # root = '/home/yangxin/datasets/3d_corn/miao_corn/transformed/20230213'

    TEST_DATASET = CornDataset(split='test', data_dir="data_pretreatment", meta=META, root=root,
                               npoints=NUM_POINT)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
                                                 pin_memory=True, drop_last=True)
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():

        MINI_LEAF = 2
        CORN_CLASS = 11

        leaf_group_Precision = [[] for _ in range(CORN_CLASS)]  # 4~11 所以是8  2~12 11
        leaf_group_Recall = [[] for _ in range(CORN_CLASS)]  # 4~11 所以是8
        # leaf_group_Accuracy = [[] for _ in range(CORN_CLASS)]
        leaf_group_F1 = [[] for _ in range(CORN_CLASS)]
        leaf_group_IoU = [[] for _ in range(CORN_CLASS)]
        leaf_group_acc = [[] for _ in range(CORN_CLASS)]

        log_string('---- EVALUATION ----')

        single_res_f = open(os.path.join(experiment_dir, 'single_res.csv'), 'w')
        single_res_f.write(f"name,Accuracy,mPrecision,mRecall,mF1,mIoU,"
                           f"stem:mPrecision,mRecall,mF1,mIoU,"
                           f"leaf:mPrecision,mRecall,mF1,mIoU\n")
        for i, (points, target, id) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            points_ = points.data.numpy()
            points = torch.Tensor(points_)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            vote_pool = torch.zeros(target.size()[0], target.size()[1], NUM_CLASSES).cuda()
            for _ in range(NUM_VOTES):
                pred, _ = classifier(points)
                vote_pool += pred
            pred = vote_pool / NUM_VOTES
            pred_val = np.argmax(pred.cpu().data.numpy(), 2)
            batch_label = target.cpu().data.numpy()

            # seg_pred, trans_feat = classifier(points)
            # pred_val = seg_pred.contiguous().cpu().data.numpy()
            # seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            # batch_label = target.cpu().data.numpy()
            # pred_val = np.argmax(pred_val, 2)

            for j in range(pred_val.shape[0]):
                one_pred_val = pred_val[j]
                one_target = batch_label[j]
                one_points = points_[j]
                one_id = id[j]
                leaf_num = int(one_id.split('-')[1]) - MINI_LEAF

                # one_Precision = [0 for _ in range(NUM_CLASSES)]
                # one_Recall = [0 for _ in range(NUM_CLASSES)]
                # one_Accuracy = [0 for _ in range(NUM_CLASSES)]
                # one_F1 = [0 for _ in range(NUM_CLASSES)]
                # one_IoU =  [0 for _ in range(NUM_CLASSES)]

                one_Precision, one_Recall, one_F1, _ = precision_recall_fscore_support(one_target, one_pred_val)
                one_IoU = compute_iou(one_target, one_pred_val)
                one_acc = accuracy_score(one_target, one_pred_val)

                # for l in range(NUM_CLASSES):
                #     TP = np.sum((one_pred_val == l) & (one_target == l))
                #     FP = np.sum((one_pred_val == l) & (one_target != l))
                #     TN = np.sum((one_pred_val != l) & (one_target != l))
                #     FN = np.sum((one_pred_val != l) & (one_target == l))
                #
                #     # one_Precision[l] = round(TP / (TP + FP), 6)
                #     # one_Recall[l] = round(TP / (TP + FN), 6)
                #     # one_Accuracy[l] = round((TP + TN) / (TP + FP + TN + FN), 6)
                #     # one_F1[l] = round(2 * one_Precision[l] * one_Recall[l] / (one_Precision[l] + one_Recall[l]), 6)
                #     one_IoU[l] = round(TP / (TP + FP + FN), 6)


                leaf_group_Precision[leaf_num].append(one_Precision)
                leaf_group_Recall[leaf_num].append(one_Recall)
                # leaf_group_Accuracy[leaf_num].append(one_Accuracy)
                leaf_group_F1[leaf_num].append(one_F1)
                leaf_group_IoU[leaf_num].append(one_IoU)
                leaf_group_acc[leaf_num].append(one_acc)
                # 测试结果保存
                if VISUAL:
                    points_pred = np.concatenate([one_points, one_pred_val.reshape(-1, 1)], axis=1)
                    points_target = np.concatenate([one_points, one_target.reshape(-1, 1)], axis=1)
                    pred_filename = one_id + '_pred.txt'
                    target_filename = one_id + '_target.txt'
                    np.savetxt(os.path.join(visual_dir, pred_filename), points_pred, fmt="%.6f")
                    np.savetxt(os.path.join(visual_dir, target_filename), points_target, fmt="%.6f")


                    with open(os.path.join(visual_dir, one_id + '_sign.txt'), 'w') as f:
                        f.write(f"class: {' ' * (14 - len('class: '))}")
                        f.write(f"Accuracy: {' ' * (14 - len('Accuracy: '))}")
                        f.write(f"Precision: {' ' * (14 - len('Precision: '))}")
                        f.write(f"Recall: {' ' * (14 - len('Recall: '))}")
                        f.write(f"Accuracy: {' ' * (14 - len('mAccuracy: '))}")
                        f.write(f"IoU: {' ' * (14 - len('mIoU: '))}")
                        f.write(f"F1: {' ' * (14 - len('mIoU: '))}\n")

                        f.write(f"total : {' ' * (14 - len('mIoU: '))}")

                        Accuracy = round(one_acc, 6)
                        mPrecision = np.round(np.mean(one_Precision), 6)
                        mRecall = np.round(np.mean(one_Recall), 6)
                        # mAccuracy = np.round(np.mean(one_Accuracy), 6)
                        mIoU = np.round(np.mean(one_IoU), 6)
                        mF1 = np.round(np.mean(one_F1), 6)

                        f.write(f"{str(Accuracy) + ' ' * (14 - len(str(Accuracy)))}")
                        f.write(f"{str(mPrecision) + ' ' * (14 - len(str(mPrecision)))}")
                        f.write(f"{str(mRecall) + ' ' * (14 - len(str(mRecall)))}")
                        # f.write(f"{str(mAccuracy) + ' ' * (14 - len(str(mAccuracy)))}")
                        f.write(f"{str(mIoU) + ' ' * (14 - len(str(mIoU)))}")
                        f.write(f"{str(mF1) + ' ' * (14 - len(str(mF1)))}\n")

                        single_res_f.write(f"{one_id},{Accuracy},{mPrecision},{mRecall},{mF1},{mIoU},"
                                           f"{one_Precision[0]},{one_Recall[0]},{one_F1[0]},{one_IoU[0]},"
                                           f"{one_Precision[1]},{one_Recall[1]},{one_F1[1]},{one_IoU[1]}\n")

                        for l in range(NUM_CLASSES):
                            f.write(f"{seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l]))}")
                            f.write(f"{' ' * 14}")
                            f.write(f"{str(one_Precision[l]) + ' ' * (14 - len(str(one_Precision[l])))}")
                            f.write(f"{str(one_Recall[l]) + ' ' * (14 - len(str(one_Recall[l])))}")
                            # f.write(f"{str(one_Accuracy[l]) + ' ' * (14 - len(str(one_Accuracy[l])))}")
                            f.write(f"{str(one_IoU[l]) + ' ' * (14 - len(str(one_IoU[l])))}")
                            f.write(f"{str(one_F1[l]) + ' ' * (14 - len(str(one_F1[l])))}\n")


        result_str = f"\nclass: {' ' * (14 - len('class: '))}" \
                     + f"Accuracy: {' ' * (14 - len('Accuracy: '))}" \
                     + f"Precision: {' ' * (14 - len('Precision: '))}" \
                     + f"Recall: {' ' * (14 - len('Recall: '))}" \
                     + f"IoU: {' ' * (14 - len('IoU: '))}" \
                     + f"F1: {' ' * (14 - len('F1: '))}\n"

        leaf_group_Acc = []
        leaf_group_mPrecision = []
        leaf_group_mRecall = []
        # leaf_group_mAccuracy = []
        leaf_group_mIoU = []
        leaf_group_mF1 = []

        f_res = open(os.path.join(experiment_dir, 'result.csv'), 'w')

        for i in range(CORN_CLASS):
            one_acc = np.round(np.mean(leaf_group_acc[i]), 6)

            one_Precision = np.round(np.mean(leaf_group_Precision[i], axis=0), 6)
            one_Recall = np.round(np.mean(leaf_group_Recall[i], axis=0), 6)
            # one_Accuracy = np.round(np.mean(leaf_group_Accuracy[i], axis=0), 6)
            one_IoU = np.round(np.mean(leaf_group_IoU[i], axis=0), 6)
            one_F1 = np.round(np.mean(leaf_group_F1[i], axis=0), 6)

            leaf_group_Acc.append(one_acc)
            leaf_group_mPrecision.append(one_Precision)
            leaf_group_mRecall.append(one_Recall)
            # leaf_group_mAccuracy.append(one_Accuracy)
            leaf_group_mIoU.append(one_IoU)
            leaf_group_mF1.append(one_F1)

            mPrecision = np.round(np.mean(one_Precision), 6)
            mRecall = np.round(np.mean(one_Recall), 6)
            # mAccuracy = np.round(np.mean(one_Accuracy), 6)
            mIoU = np.round(np.mean(one_IoU), 6)
            mF1 = np.round(np.mean(one_F1), 6)

            m_class_str = str(i + MINI_LEAF) + ': avg'
            result_str += f"{m_class_str + ' ' * (14 - len(m_class_str))}" \
                  + f"{str(one_acc) + ' ' * (14 - len(str(one_acc)))}" \
                  + f"{str(mPrecision) + ' ' * (14 - len(str(mPrecision)))}" \
                  + f"{str(mRecall) + ' ' * (14 - len(str(mRecall)))}" \
                  + f"{str(mIoU) + ' ' * (14 - len(str(mIoU)))}" \
                  + f"{str(mF1) + ' ' * (14 - len(str(mF1)))}\n"

            for l in range(NUM_CLASSES):
                class_str = str(i + MINI_LEAF) + ': ' + seg_label_to_cat[l]
                result_str += f"{class_str + ' ' * (14 - len(class_str))}" \
                     + f"{' ' * 14}" \
                     + f"{str(one_Precision[l]) + ' ' * (14 - len(str(one_Precision[l])))}" \
                     + f"{str(one_Recall[l]) + ' ' * (14 - len(str(one_Recall[l])))}" \
                     + f"{str(one_IoU[l]) + ' ' * (14 - len(str(one_IoU[l])))}" \
                     + f"{str(one_F1[l]) + ' ' * (14 - len(str(one_F1[l])))}\n"

                f_res.write(f"{i + MINI_LEAF},{l},{one_Precision[l]},{one_Recall[l]},{one_IoU[l]},{one_F1[l]}\n")


        total_acc = np.round(np.mean(leaf_group_Acc), 6)
        total_Precision = np.round(np.mean(leaf_group_mPrecision, axis=0), 6)
        total_Recall = np.round(np.mean(leaf_group_mRecall, axis=0), 6)
        # total_Accuracy = np.round(np.mean(leaf_group_mAccuracy, axis=0), 6)
        total_IoU = np.round(np.mean(leaf_group_mIoU, axis=0), 6)
        total_F1 = np.round(np.mean(leaf_group_mF1, axis=0), 6)

        total_mPrecision = np.round(np.mean(total_Precision), 6)
        total_mRecall = np.round(np.mean(total_Recall), 6)
        # total_mAccuracy = np.round(np.mean(total_Accuracy), 6)
        total_mIoU = np.round(np.mean(total_IoU), 6)
        total_mF1 = np.round(np.mean(total_F1), 6)


        m_class_str = 'total: '
        result_str += f"\n{m_class_str + ' ' * (14 - len(m_class_str))}" \
                      + f"{str(total_acc) + ' ' * (14 - len(str(total_acc)))}" \
                      + f"{str(total_mPrecision) + ' ' * (14 - len(str(total_mPrecision)))}" \
                      + f"{str(total_mRecall) + ' ' * (14 - len(str(total_mRecall)))}" \
                      + f"{str(total_mIoU) + ' ' * (14 - len(str(total_mIoU)))}" \
                      + f"{str(total_mF1) + ' ' * (14 - len(str(total_mF1)))}\n"

        f_res.write(f"{-1},{total_acc},{total_mPrecision},{total_mRecall},{total_mIoU},{total_mF1}\n")

        for l in range(NUM_CLASSES):
            class_str = seg_label_to_cat[l]
            result_str += f"{class_str + ' ' * (14 - len(class_str))}" \
                          + f"{' ' * 14}" \
                          + f"{str(total_Precision[l]) + ' ' * (14 - len(str(total_Precision[l])))}" \
                          + f"{str(total_Recall[l]) + ' ' * (14 - len(str(total_Recall[l])))}" \
                          + f"{str(total_IoU[l]) + ' ' * (14 - len(str(total_IoU[l])))}" \
                          + f"{str(total_F1[l]) + ' ' * (14 - len(str(total_F1[l])))}\n"
            f_res.write(f"{l},{total_Precision[l]},{total_Recall[l]},{total_IoU[l]},{total_F1[l]}\n")

        log_string(result_str)
        f_res.close()
        single_res_f.close()


        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
