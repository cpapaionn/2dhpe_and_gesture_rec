from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision

from core.inference import get_final_preds_enhanced
from utils.transforms import flip_back


def predict(cfg, model, img, bbox, device=None):

    aspect_ratio = cfg.image_width * 1.0 / cfg.image_height

    test_img, fixed_bbox = crop_img_to_bbox(img, bbox, aspect_ratio_thres=aspect_ratio)

    normalize = torchvision.transforms.Normalize(mean=cfg.image_mean, std=cfg.image_std)
    input_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])

    test_img = input_transform(test_img)
    test_img = test_img.unsqueeze(0)
    test_img = F.interpolate(test_img, size=(cfg.image_height, cfg.image_width), mode='bilinear', align_corners=True)

    model.eval()

    with torch.no_grad():
        test_img = test_img.to(device)

        # compute output
        output = model(test_img)

        if cfg.FLIP_TEST:
            test_img_flipped = test_img.flip(3)
            output_flipped = model(test_img_flipped)

            output_flipped = flip_back(output_flipped.cpu().numpy(), cfg.FLIP_PAIRS)
            output_flipped = torch.from_numpy(output_flipped.copy()).to(device)

            # feature is not aligned, shift flipped heatmap for higher accuracy
            if cfg.SHIFT_HEATMAP:
                output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5

    c, s = bbox2cs(fixed_bbox, aspect_ratio_thres=aspect_ratio)
    preds, maxvals = get_final_preds_enhanced(cfg, output.clone().cpu().numpy(), c, s)

    return np.squeeze(preds), np.squeeze(maxvals)


def crop_img_to_bbox(img, bbox, aspect_ratio_thres=0.75):
    # Adjust bbox for 2D human pose model
    bbox_y, bbox_x, bbox_h, bbox_w = bbox[0], bbox[1], bbox[2], bbox[3]
    if bbox_y < 0:
        bbox_y = 0
    if bbox_x < 0:
        bbox_x = 0
    
    bbox_x_c, bbox_y_c, bbox_h_c, bbox_w_c = bbox_x, bbox_y, bbox_h, bbox_w

    if bbox_w / float(bbox_h) < aspect_ratio_thres:
        bbox_h_new = int(bbox_h)
        bbox_w_new = int(bbox_h * aspect_ratio_thres)
        bbox_x_new = int(bbox_x - ((bbox_w_new - bbox_w) // 2))
        if bbox_x_new < 0:
            bbox_x_new = 0
        if bbox_x_new + bbox_w_new > img.shape[1]:
            bbox_w_new = img.shape[1] - bbox_x_new
            bbox_h_new = int(bbox_w_new / aspect_ratio_thres)
        bbox_x_c, bbox_y_c, bbox_h_c, bbox_w_c = bbox_x_new, bbox_y, bbox_h_new, bbox_w_new
    elif bbox_w / float(bbox_h) > aspect_ratio_thres:
        bbox_w_new = int(bbox_w)
        bbox_h_new = int(bbox_w / aspect_ratio_thres)
        bbox_y_new = int(bbox_y - ((bbox_h_new - bbox_h) // 2))
        if bbox_y_new < 0:
            bbox_y_new = 0
        if bbox_y_new + bbox_h_new > img.shape[0]:
            bbox_h_new = img.shape[0] - bbox_y_new
            bbox_w_new = int(bbox_h_new * aspect_ratio_thres)
        bbox_x_c, bbox_y_c, bbox_h_c, bbox_w_c = bbox_x, bbox_y_new, bbox_h_new, bbox_w_new

    cropped_img = img[bbox_y_c:bbox_y_c+bbox_h_c, bbox_x_c:bbox_x_c+bbox_w_c]

    return cropped_img, [bbox_y_c, bbox_x_c, bbox_h_c, bbox_w_c]


def bbox2cs(bbox, aspect_ratio_thres=0.75, pixel_std=200):
    y, x, h, w = bbox[:]

    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio_thres * h:
        h = w * 1.0 / aspect_ratio_thres
    elif w < aspect_ratio_thres * h:
        w = h * aspect_ratio_thres

    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    # if center[0] != -1:
    #     scale = scale * 1.25

    return np.expand_dims(center, axis=0), np.expand_dims(scale, axis=0)
