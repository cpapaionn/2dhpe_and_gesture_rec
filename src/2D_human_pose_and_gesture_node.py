#!/usr/bin/env python3
from __future__ import print_function, division

import os
import sys
import numpy as np
import cv2

import roslib
import rospy, rospkg
from sensor_msgs.msg import Image
from visualanalysis_msgs.msg import ROI2D, TargetLocations2D, BodyJoint2D, BodyJoint2DArray, Gesture
from cv_bridge import CvBridge, CvBridgeError

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision

from HumanPose2D.config import config as cfg
from HumanPose2D.network import CNN_GAN_AS
from core.function import predict

from Gesture.network import GestureRec
from Gesture.utils import predict_gesture

from subprocess import call

class Human_Pose_2D_and_Gesture_node:


	def __init__(self):

		# Cuda settings
		cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.enabled = True
		self.device = ("cuda" if torch.cuda.is_available() else "cpu")

		# set paths
		self.rospack = rospkg.RosPack()
		self.base_path = self.rospack.get_path("visualanalysis_acw")
		self.dir_path = os.path.dirname(os.path.realpath(__file__))
		common_path = self.rospack.get_path("ac_tools")
		
		self.model_2d_weights_path = rospy.get_param("pose2d_model_weights_path", "data/models/HumanPoseWeights/model_2d.pth")
		self.pose_2d_model_weights_path = os.path.join(self.base_path, self.model_2d_weights_path)
		# check if paths available
		if not os.path.exists(self.pose_2d_model_weights_path):
			command = os.path.join(common_path, 'scripts/gdrive.sh') + ' ' + rospy.get_param("gdrive_url_pose2d") + ' ' + self.pose_2d_model_weights_path
			dc = call(command, shell=True)
			
		self.gesture_class_num = rospy.get_param("~gesture_class_num", 5)
		
		if self.gesture_class_num == 5:	
			self.model_gesture_weights_path = rospy.get_param("gesture_model_5g_weights_path", "data/models/GestureWeights/mixed_5g_ref.pth")
			self.gesture_model_weights_path = os.path.join(self.base_path, self.model_gesture_weights_path)
			# check if paths available
			if not os.path.exists(self.gesture_model_weights_path):
				command = os.path.join(common_path, 'scripts/gdrive.sh') + ' ' + rospy.get_param("gdrive_url_gesture_5classes") + ' ' + self.gesture_model_weights_path
				dc = call(command, shell=True)
		elif self.gesture_class_num == 6:
			self.model_gesture_weights_path = rospy.get_param("gesture_model_6g_weights_path", "data/models/GestureWeights/mixed_6g_ref.pth")
			self.gesture_model_weights_path = os.path.join(self.base_path, self.model_gesture_weights_path)
			# check if paths available
			if not os.path.exists(self.gesture_model_weights_path):
				command = os.path.join(common_path, 'scripts/gdrive.sh') + ' ' + rospy.get_param("gdrive_url_gesture_6classes") + ' ' + self.gesture_model_weights_path
				dc = call(command, shell=True)
		

		# init models
		self.model_pose_2d = CNN_GAN_AS(cfg.num_classes, is_training=False, norm_layer=torch.nn.BatchNorm2d)
		self.model_pose_2d = torch.nn.DataParallel(self.model_pose_2d)
		self.model_pose_2d.load_state_dict(torch.load(self.pose_2d_model_weights_path), strict=False)
		self.model_pose_2d.to(self.device)
		
		self.model_gesture = GestureRec(output_size=self.gesture_class_num, device=self.device)
		self.model_gesture.load_state_dict(torch.load(self.gesture_model_weights_path), strict=False)
		self.model_gesture.to(self.device)
		self.model_gesture.eval()

		# inits
		self.detections = None
		self.gesture_duration = 9
		self.gesture_prediction_interval = int(rospy.get_param("gesture_prediction_interval", 1))
		self.no_gesture_thres = float(rospy.get_param("no_gesture_thres", 0.2))
		self.frame_counter = 0
		self.frame_buffer = []
		self.gesture_buffer = []
		self.gesture_buffer_size = 18

		self.bridge = CvBridge()
		# self.img_sub_topic = rospy.get_param("~camera_topic", "aerial_coworker/visualanalysis/rgb_camera")
		# self.img_sub = rospy.Subscriber(self.img_sub_topic, Image, self.get_image, queue_size=10)
		self.img_sub_topic = rospy.get_param("~det_img_topic")
		self.img_sub = rospy.Subscriber(self.img_sub_topic, Image, self.predict_human_pose_2d_and_gesture, queue_size=1)
		self.det_topic = rospy.get_param("~det_topic")
		self.sub = rospy.Subscriber(self.det_topic, TargetLocations2D, self.get_detections, queue_size=1)
		self.pub_topic = rospy.get_param("~pose_2d_topic")
		self.pub = rospy.Publisher(self.pub_topic, BodyJoint2DArray, queue_size=1)
		self.pose_img_pub_topic = rospy.get_param("~pose_img_topic")
		self.pose_img_pub = rospy.Publisher(self.pose_img_pub_topic, Image, queue_size=1)
		self.pose_det_pub_topic = rospy.get_param("~pose_det_topic")
		self.pose_det_pub = rospy.Publisher(self.pose_det_pub_topic, TargetLocations2D, queue_size=1)
		self.gesture_pub_topic = rospy.get_param("~gesture_topic")
		self.gesture_pub = rospy.Publisher(self.gesture_pub_topic, Gesture, queue_size=1)
		
		rospy.loginfo('2D Human Pose-Gesture Node initiated')


	# def get_image(self, img_msg):
	# 	self.person_img_msg = img_msg
	
	
	def get_detections(self, data):
		self.detections = []
		detections = data.detections
		for detection in detections:
			if detection.class_id != 1:
				continue

			bbox_x = detection.x
			bbox_y = detection.y
			bbox_h = detection.h
			bbox_w = detection.w
			bbox_det_score = detection.det_score
			bbox_class = detection.class_id
			self.detections.append([bbox_x, bbox_y, bbox_h, bbox_w, bbox_class, bbox_det_score])
		

	def predict_human_pose_2d_and_gesture(self, img_msg):

		if self.detections is not None and len(self.detections) > 0:
		
			frame = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
			detections = np.array(self.detections)

			for s in range(0, detections.shape[0]):

				if detections[s][-2] != 1:
					continue    
		
				bbox_x = detections[s][0]
				bbox_y = detections[s][1]
				bbox_h = detections[s][2]
				bbox_w = detections[s][3]
				bbox_det_score = detections[s][-1]
				
				# extra_w = bbox_w * 1.5
				# extra_h = bbox_h * 0.3
				# bbox_x = bbox_x - (extra_w / 2.)
				# if bbox_x < 0:
				# 	bbox_x = 0
				# bbox_y = bbox_y - (extra_h / 2.)
				# if bbox_y < 0:
				# 	bbox_y = 0
				# bbox_w = bbox_w + (extra_w / 2.)
				# if (bbox_x + bbox_w) > frame.shape[1]:
				# 	bbox_w = (frame.shape[1] - bbox_x) - 1
				# bbox_h = bbox_h + (extra_h / 2.)
				# if (bbox_y + bbox_h) > frame.shape[0]:
				# 	bbox_h = (frame.shape[0] - bbox_y) - 1
				
				test_bbox = np.array([bbox_y, bbox_x, bbox_h, bbox_w]).astype(int)
				
				joints, confidence = predict(cfg, self.model_pose_2d, frame, test_bbox, self.device)
				
				self.frame_buffer.append(joints)
				self.frame_counter += 1

				if self.frame_counter == self.gesture_duration:
					gesture_id, gest_confidence = predict_gesture(self.model_gesture, np.array(self.frame_buffer), 
															 	  gesture_class_num=self.gesture_class_num, no_gesture_thres=self.no_gesture_thres, 
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

					gesture = Gesture()
					gesture.gesture_id = gesture_id + 1
					gesture.score = gest_confidence
					
					self.gesture_pub.publish(gesture) 
					rospy.loginfo('Gesture published: %i', gesture_id + 1)
					
					self.frame_counter -= self.gesture_prediction_interval
					for ii in range(self.gesture_prediction_interval):
						self.frame_buffer.pop(0)

				# publish detected 2D skeleton and scores
				skeleton2d = BodyJoint2DArray()
				# skeleton2d.frame_id = frame_id
				skeleton2darr = []

				for idx in range(0, joints.shape[0]):
					joint2d = BodyJoint2D()
					joint2d.joint_id = idx
					joint2d.x = int(joints[idx, 0])
					joint2d.y = int(joints[idx, 1])
					joint2d.score = confidence[idx]
					skeleton2darr.append(joint2d)
				skeleton2d.skeleton_2d = skeleton2darr
				
				tgt_msg = TargetLocations2D()
				roi_msg = ROI2D()
				roi_msg.roi_id = 0
				roi_msg.x = int(bbox_x)
				roi_msg.y = int(bbox_y)
				roi_msg.w = int(bbox_w)
				roi_msg.h = int(bbox_h)
				roi_msg.det_score = bbox_det_score
				roi_msg.class_id = int(detections[s][-2])
				tgt_msg.detections.append(roi_msg)

				self.pub.publish(skeleton2d)
				rospy.loginfo('2D skeleton published')
				self.pose_det_pub.publish(tgt_msg)
				self.pose_img_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

		

def main(args):
	rospy.init_node('Human_Pose_2D_and_Gesture', anonymous=True)
	hpose2d = Human_Pose_2D_and_Gesture_node()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	return 0


if __name__ == '__main__':
	main(sys.argv)


