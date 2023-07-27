from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import cv2


def visualize_2D_skeleton(img, skeleton_2d, confidence, confidence_thres=0.):

    weak_preds = confidence < confidence_thres
    for idx, cut_out in enumerate(weak_preds):
        if cut_out:
            skeleton_2d[idx, :] = [-1, -1]

    if np.all(skeleton_2d[16] != [-1, -1]) and np.all(skeleton_2d[14] != [-1, -1]):
        cv2.line(img, (int(skeleton_2d[16][0]), int(skeleton_2d[16][1])), (int(skeleton_2d[14][0]), int(skeleton_2d[14][1])), (255, 0, 0), 3)
    if np.all(skeleton_2d[14] != [-1, -1]) and np.all(skeleton_2d[12] != [-1, -1]):
        cv2.line(img, (int(skeleton_2d[14][0]), int(skeleton_2d[14][1])), (int(skeleton_2d[12][0]), int(skeleton_2d[12][1])), (255, 234, 0), 3)
    if np.all(skeleton_2d[12] != [-1, -1]) and np.all(skeleton_2d[6] != [-1, -1]):
        cv2.line(img, (int(skeleton_2d[12][0]), int(skeleton_2d[12][1])), (int(skeleton_2d[6][0]), int(skeleton_2d[6][1])), (187, 255, 0), 3)
    if np.all(skeleton_2d[6] != [-1, -1]) and np.all(skeleton_2d[8] != [-1, -1]):
        cv2.line(img, (int(skeleton_2d[6][0]), int(skeleton_2d[6][1])), (int(skeleton_2d[8][0]), int(skeleton_2d[8][1])), (255, 0, 242), 3)
    if np.all(skeleton_2d[8] != [-1, -1]) and np.all(skeleton_2d[10] != [-1, -1]):
         cv2.line(img, (int(skeleton_2d[8][0]), int(skeleton_2d[8][1])), (int(skeleton_2d[10][0]), int(skeleton_2d[10][1])), (132, 0, 255), 3)
    if np.all(skeleton_2d[6] != [-1, -1]) and np.all(skeleton_2d[5] != [-1, -1]):
         cv2.line(img, (int(skeleton_2d[6][0]), int(skeleton_2d[6][1])), (int(skeleton_2d[5][0]), int(skeleton_2d[5][1])), (0, 255, 0), 3)
    if np.all(skeleton_2d[11] != [-1, -1]) and np.all(skeleton_2d[13] != [-1, -1]):
         cv2.line(img, (int(skeleton_2d[11][0]), int(skeleton_2d[11][1])), (int(skeleton_2d[13][0]), int(skeleton_2d[13][1])), (0, 255, 217), 3)
    if np.all(skeleton_2d[13] != [-1, -1]) and np.all(skeleton_2d[15] != [-1, -1]):
         cv2.line(img, (int(skeleton_2d[13][0]), int(skeleton_2d[13][1])), (int(skeleton_2d[15][0]), int(skeleton_2d[15][1])), (0, 204, 255), 3)
    if np.all(skeleton_2d[0] != [-1, -1]) and np.all(skeleton_2d[2] != [-1, -1]):
         cv2.line(img, (int(skeleton_2d[0][0]), int(skeleton_2d[0][1])), (int(skeleton_2d[2][0]), int(skeleton_2d[2][1])), (255, 132, 130), 3)
    if np.all(skeleton_2d[11] != [-1, -1]) and np.all(skeleton_2d[5] != [-1, -1]):
         cv2.line(img, (int(skeleton_2d[11][0]), int(skeleton_2d[11][1])), (int(skeleton_2d[5][0]), int(skeleton_2d[5][1])), (255, 0, 166), 3)
    if np.all(skeleton_2d[5] != [-1, -1]) and np.all(skeleton_2d[7] != [-1, -1]):
         cv2.line(img, (int(skeleton_2d[5][0]), int(skeleton_2d[5][1])), (int(skeleton_2d[7][0]), int(skeleton_2d[7][1])), (169, 212, 110), 3)
    if np.all(skeleton_2d[7] != [-1, -1]) and np.all(skeleton_2d[9] != [-1, -1]):
         cv2.line(img, (int(skeleton_2d[7][0]), int(skeleton_2d[7][1])), (int(skeleton_2d[9][0]), int(skeleton_2d[9][1])), (156, 240, 202), 3)
    if np.all(skeleton_2d[1] != [-1, -1]) and np.all(skeleton_2d[0] != [-1, -1]):
         cv2.line(img, (int(skeleton_2d[1][0]), int(skeleton_2d[1][1])), (int(skeleton_2d[0][0]), int(skeleton_2d[0][1])), (156, 238, 240), 3)
    if np.all(skeleton_2d[1] != [-1, -1]) and np.all(skeleton_2d[3] != [-1, -1]):
         cv2.line(img, (int(skeleton_2d[1][0]), int(skeleton_2d[1][1])), (int(skeleton_2d[3][0]), int(skeleton_2d[3][1])), (186, 186, 255), 3)
    if np.all(skeleton_2d[2] != [-1, -1]) and np.all(skeleton_2d[4] != [-1, -1]):
         cv2.line(img, (int(skeleton_2d[2][0]), int(skeleton_2d[2][1])), (int(skeleton_2d[4][0]), int(skeleton_2d[4][1])), (186, 106, 106), 3)

    for k in range(0, skeleton_2d.shape[0]):
        joint = skeleton_2d[k]
        if np.all(joint != [-1, -1]):
            cv2.circle(img, (int(joint[0]), int(joint[1])), 4, [0, 0, 0], 1)
            # cv2.putText(img, str(k), (int(joint[0]), int(joint[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return img

      

