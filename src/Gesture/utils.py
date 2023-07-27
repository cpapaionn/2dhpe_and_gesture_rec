import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

def select_joints(sequence):
    """
    This method isolates the nose and upper body limb joints.
    :param sequence: The input sequence.
    :return: The sequence with the isolated joints.
    """
    return sequence[:, [0, 5, 6, 7, 8, 9, 10, 17, 22, 23, 24, 25, 26, 27]]


def preprocess_pose(sequence):
    """
    This method applies pre-processing transformations to an action sequence.
    :param sequence: The input sequence.
    :return: The processed sequence.
    """
    out = np.copy(sequence)
    for i in range(out.shape[0]):
        out[i, :out.shape[1] // 2] -= np.mean(out[i, :out.shape[1] // 2])
        out[i, out.shape[1] // 2:] -= np.mean(out[i, out.shape[1] // 2:])

        out[i] /= np.std(out[i])

    return out
    
    
def entropy(inp):
	h = F.softmax(inp, dim=1) * F.log_softmax(inp, dim=1)
	h = -1.0 * h.sum(dim=1)

	return h


def predict_gesture(model, sequence, gesture_class_num=5, no_gesture_thres=100., postp=False, device=None):
	"""
    This method gets an action sequence with shape (9, 17, 2)
    [batch_size, sequence_size, features], with the features
    formed as (x1, x2, ..., x17, y1, y2, ..., y17),
    and recognises the performed gesture.
    :param sequence: the action sequence.
    :param return_label: A boolean which if set to 'True' will
                        make the function return a string label
                        instead of an integer label.
	:return: the label of the action
	"""

	# Data processing.
	sequence = np.hstack([sequence[:, :, 0], sequence[:, :, 1]])
	sequence = select_joints(sequence)
	sequence = preprocess_pose(sequence)
	sequence = torch.tensor(sequence, dtype=torch.float32, device=device)

    # Inference.
	with torch.no_grad():
		output = F.log_softmax(model(sequence.unsqueeze(0)), dim=1)
		output_np = output.detach().cpu().numpy()[0]
		if postp:
			ent = entropy(output)
			# print(ent)
			#if ent > no_gesture_thres:
			#	gesture_id = gesture_class_num
			#	confidence = 0.
			#else:
			gesture_id = np.argmax(output_np)
			confidence = np.max(output_np)
			if gesture_id == 4 and ent > no_gesture_thres / 100.:
				gesture_id = gesture_class_num
				confidence = 0.
		else:
			gesture_id = np.argmax(output_np)
			confidence = np.max(output_np)
			
	return gesture_id, confidence
    
