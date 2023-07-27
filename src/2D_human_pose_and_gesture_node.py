from __future__ import print_function, division

import numpy as np
import yaml

import cv2
import torch

from dtr_framework import build_detector

from HumanPose2D.config import config as cfg_hp2d
from HumanPose2D.network import CNN_GAN_AS
from core.function import predict as predict_hp2d

from Gesture.network import GestureRec
from Gesture.utils import predict_gesture


class Human_Pose_2D_and_Gesture_predictor:
	def __init__(self, cfg):
		self.device = ("cuda" if torch.cuda.is_available() else "cpu")
		
		self.pose_2d_model_weights_path = cfg['pose2d_model_weights_path']

		self.gesture_class_num = cfg['gesture_class_num']
		if self.gesture_class_num == 5:	
			self.gesture_model_weights_path = cfg['gesture_model_5g_weights_path']
		elif self.gesture_class_num == 6:
			self.gesture_model_weights_path = cfg['gesture_model_6g_weights_path']

		# init models
		self.model_person_detection = build_detector(cfg)

		self.model_pose_2d = CNN_GAN_AS(cfg_hp2d.num_classes, is_training=False, norm_layer=torch.nn.BatchNorm2d)
		self.model_pose_2d = torch.nn.DataParallel(self.model_pose_2d)
		self.model_pose_2d.load_state_dict(torch.load(self.pose_2d_model_weights_path, map_location=torch.device(self.device)), strict=False)
		self.model_pose_2d.to(self.device)
		
		self.model_gesture = GestureRec(output_size=self.gesture_class_num, device=self.device)
		self.model_gesture.load_state_dict(torch.load(self.gesture_model_weights_path, map_location=torch.device(self.device)), strict=False)
		self.model_gesture.to(self.device)
		self.model_gesture.eval()

		# inits
		self.detections = None
		self.gesture_duration = 9
		self.gesture_prediction_interval = cfg['gesture_prediction_interval']
		self.no_gesture_thres = cfg['no_gesture_thres']
		self.frame_counter = 0
		self.frame_buffer = []
		self.gesture_buffer = []
		self.gesture_buffer_size = 18

	# def get_detections(self, data):
	# 	self.detections = []
	# 	detections = data.detections
	# 	for detection in detections:
	# 		if detection.class_id != 1:
	# 			continue
	#
	# 		bbox_x = detection.x
	# 		bbox_y = detection.y
	# 		bbox_h = detection.h
	# 		bbox_w = detection.w
	# 		bbox_det_score = detection.det_score
	# 		bbox_class = detection.class_id
	# 		self.detections.append([bbox_x, bbox_y, bbox_h, bbox_w, bbox_class, bbox_det_score])

	def predict(self, img_path):
		frame = cv2.imread(img_path)
		detections = self.model_person_detection.detect(frame)
		detections = np.array(detections)
		for s in range(0, detections.shape[0]):
			if detections[s][-1] != 1:
				continue

			bbox_x = detections[s][0]
			bbox_y = detections[s][1]
			bbox_w = detections[s][2]
			bbox_h = detections[s][3]
			bbox_det_score = detections[s][-2]

			test_bbox = np.array([bbox_y, bbox_x, bbox_h, bbox_w]).astype(int)

			joints, confidence = predict_hp2d(cfg_hp2d, self.model_pose_2d, frame, test_bbox, self.device)

			self.frame_buffer.append(joints)
			self.frame_counter += 1

			if self.frame_counter == self.gesture_duration:
				gesture_id, gest_confidence = predict_gesture(self.model_gesture,
															np.array(self.frame_buffer),
															gesture_class_num=self.gesture_class_num,
															no_gesture_thres=self.no_gesture_thres,
															postp=True, device=self.device)

				self.gesture_buffer.append(gesture_id)
				if len(self.gesture_buffer) < self.gesture_buffer_size:
					gesture_id = self.gesture_class_num
					gest_confidence = 0.
				else:
					all_same = all([g_id==gesture_id for g_id in self.gesture_buffer])
					if not all_same:
						gesture_id = self.gesture_class_num
						gest_confidence = 0.
					self.gesture_buffer.pop(0)

				gesture_id += 1
				score = gest_confidence

				print(f'Gesture published: {gesture_id}')

				self.frame_counter -= self.gesture_prediction_interval
				for ii in range(self.gesture_prediction_interval):
					self.frame_buffer.pop(0)

			# # publish detected 2D skeleton and scores
			# skeleton2d = BodyJoint2DArray()
			# # skeleton2d.frame_id = frame_id
			# skeleton2darr = []
			#
			# for idx in range(0, joints.shape[0]):
			# 	joint2d = BodyJoint2D()
			# 	joint2d.joint_id = idx
			# 	joint2d.x = int(joints[idx, 0])
			# 	joint2d.y = int(joints[idx, 1])
			# 	joint2d.score = confidence[idx]
			# 	skeleton2darr.append(joint2d)
			# skeleton2d.skeleton_2d = skeleton2darr
			#
			# tgt_msg = TargetLocations2D()
			# roi_msg = ROI2D()
			# roi_msg.roi_id = 0
			# roi_msg.x = int(bbox_x)
			# roi_msg.y = int(bbox_y)
			# roi_msg.w = int(bbox_w)
			# roi_msg.h = int(bbox_h)
			# roi_msg.det_score = bbox_det_score
			# roi_msg.class_id = int(detections[s][-2])
			# tgt_msg.detections.append(roi_msg)

		return joints, gesture_id


def main():
	with open("config/gesture_rosbag.yaml") as f:
		config = yaml.load(f, Loader=yaml.FullLoader)
	f.close()

	predictor = Human_Pose_2D_and_Gesture_predictor(config)
	test_frame = "./test_img.jpg"
	hp2d, gest_id = predictor.predict(test_frame)

	return 0


if __name__ == '__main__':
	main()


