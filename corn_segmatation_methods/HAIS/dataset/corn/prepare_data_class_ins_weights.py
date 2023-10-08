import math
import numpy as np
import glob
import os
import torch

files = sorted(glob.glob('train' + '/*.pth'))
numclass = 2
max_ins_num = 11
sem_count = [0, 0]
ins_count = [i for i in range(max_ins_num+1)]


for file in files:
    coords, colors, sem_labels, instance_labels = torch.load(file)
    sem_count[0] += np.sum(sem_labels == 0)
    sem_count[1] += np.sum(sem_labels == 1)

    max_sem = np.max(sem_labels).astype(int) + 1
    for i in range(max_sem):
        sem_count[i] += np.sum(sem_labels == i)

    max_ins = np.max(instance_labels).astype(int) + 1
    for i in range(max_ins):
        ins_count[i] += np.sum(instance_labels == i)


np.set_printoptions(suppress=True)


class_weights = np.max(sem_count) / sem_count
ins_weights = np.max(ins_count) / ins_count

print(class_weights)

ins_weights_str = ""
for i in ins_weights:
    ins_weights_str += f"{round(i, 3)}, "

print(ins_weights_str)