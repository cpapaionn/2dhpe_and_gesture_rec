# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

C.root_dir = os.path.dirname(os.path.abspath(__file__))

C.seed = 12345

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.join(C.root_dir, 'lib'))

"""Image Config"""
C.num_classes = 17
C.image_mean = [0.485, 0.456, 0.406]
C.image_std = [0.229, 0.224, 0.225]
C.image_height = 384
C.image_width = 288

""" Settings for network, this would be different for each kind of model"""
C.fix_bias = True
C.fix_bn = False
C.sync_bn = True
C.bn_eps = 1e-5
C.bn_momentum = 0.1

C.MODEL_OUTPUT_SCALE = 2
C.MODEL_DEEP = False

"""Human Pose Configs"""
C.NUM_JOINTS = 17
C.COLOR_LABELS = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                  [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 220],
                  [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                  [20, 60, 100], [50, 80, 100], [100, 100, 230], [119, 11, 32], [32, 11, 119]]

C.FLIP_TEST = True
C.FLIP_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
C.POST_PROCESS = True
C.SHIFT_HEATMAP = True
C.TEST_BLUR_KERNEL = 11
C.GPUS = []

