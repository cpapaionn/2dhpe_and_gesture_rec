import cv2
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(1)

from SiamRPN.run_SiamRPN import *
from SiamRPN.tracker_utils import rect_2_cxy_wh


class DetectorTracker(object):
    def __init__(self, cfg, det_model_path, track_model_path):
    
        # detection interval
        self.detection_interval = cfg['detection_interval']
        self.cntr = 0
        self.bboxes = []

        # detector
        self.detector = ObjectDetector(det_model_path)

        # tracker
        self.tracker = Tracker(track_model_path)
        self.tracker_initialized = False

    def detect(self, image):
        if not self.tracker_initialized:
            bboxes = self.detector.get_detections(image)

            if len(bboxes) > 0:
                rect = np.asarray(bboxes[0][:4])
                self.tracker.init_tracking(image, rect)
                self.tracker_initialized = True
        else:
            bboxes, score = self.tracker.track(image)
            if score < 0.9 or self.cntr > self.detection_interval:
                self.tracker_initialized = False
                self.cntr = 0
                bboxes = self.detector.get_detections(image)

            self.cntr += 1

        return bboxes


class ObjectDetector(object):
    def __init__(self, model_path):
        self.image = None
        self.session = None
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
                self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=self.detection_graph)

    def run_inference(self, image):
        self.image = image
        # Get handles to input and output tensors
        ops = self.session.graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self.session.graph.get_tensor_by_name(tensor_name)
        image_tensor = self.session.graph.get_tensor_by_name('image_tensor:0')
        # Run inference
        output_dict = self.session.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(self.image, 0)})
        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        return output_dict

    def get_detections(self, image):
        self.image = image
        iheight, iwidth = (self.image.shape[0], self.image.shape[1])

        output_dict = self.run_inference(self.image)

        scores = output_dict['detection_scores']
        boxes = output_dict['detection_boxes']
        classes = output_dict['detection_classes']
        
        bboxes = []

        for i in range(len([s for s in scores if s > 0.7])):
            xmin = int(boxes[i][1] * iwidth)
            ymin = int(boxes[i][0] * iheight)
            xmax = int(boxes[i][3] * iwidth)
            ymax = int(boxes[i][2] * iheight)
            width = xmax - xmin
            height = ymax - ymin

            if classes[i] == 1:
                bboxes.append([xmin, ymin, width, height, scores[i], classes[i]])
            elif classes[i] == 2:  # 62
                pass
            elif classes[i] == 3:  # 72
                pass

        return bboxes


class Tracker:
    def __init__(self, model_path):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = get_model(model_path, self.device)
        self.scale = 1.
        self.state = {}
        # self.track = False
        self.score = 0.0
        self.cfg = TrackerConfig()
        self.cfg.update(self.model.cfg)
        self.img = None
        self.initialized = False
        self.init_w_img = 1
        self.tracker_states = {}
        self.target_pos = None
        self.target_sz = None

    def init_tracking(self, img, rect):
        self.scale = 360.0 / img.shape[0]
        self.img = cv2.resize(img, (0, 0), fx=self.scale, fy=self.scale)   

        rect = rect * self.scale
        self.target_pos, self.target_sz = rect_2_cxy_wh(rect)
        self.state = SiamRPN_init(self.img, self.target_pos, self.target_sz, self.model, self.cfg, self.device)

    def track(self, img):
        bboxes = []
        iheight, iwidth = (img.shape[0], img.shape[1])
        a = 1
        img = cv2.resize(img, (0, 0), fx=self.scale, fy=self.scale)

        self.state = SiamRPN_track(self.state, img, self.device)
        x, y = int(self.state['target_pos'][0] - self.state['target_sz'][0] * 0.5), \
                             int(self.state['target_pos'][1] -
                                 self.state['target_sz'][1] * 0.5)
        w, h = int(self.state['target_sz'][0]), int(self.state['target_sz'][1])

        self.score = self.state['score']
        x, y, w, h = int(x/self.scale), int(y/self.scale), int(w/self.scale), int(h/self.scale)

        bboxes.append([x, y, w, h, self.score, 1])

        return bboxes, self.score


def build_detector(cfg):
    person_detection_model_weights_path = cfg['detection_model_name']
    tracking_model_weights_path = cfg['tracking_model_name']

    detector_tracker = DetectorTracker(cfg, person_detection_model_weights_path, tracking_model_weights_path)

    return detector_tracker
