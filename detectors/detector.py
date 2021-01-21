"""
Object detection in a single frame using YOLOv3 algorithm
Class YOLOFrameDetector processes a given frame and draws bounding boxes with detected class name and confidence.
If requested, return a JSON object with the number of detections per detected class.

Author: Carlos Cuevas
@cavesdev
February 2020
"""

import cv2 as cv
import numpy as np


class YOLOFrameDetector:
    """Class to detect objects in a given frame using YOLOv3 algorithm"""

    def __init__(self, config):
        """
        Set the necessary variables.
        :param config: is a Config object. (from config import Config)
        """
        self.confidence_threshold = config.get('confidence-threshold')  # Confidence threshold
        self.nms_threshold = config.get('nms-threshold')  # Non-maximum suppression threshold
        self.input_width = config.get('input-width')  # Width of network's input image
        self.input_height = config.get('input-height')  # Height of network's input image

        model_cfg = config.get('model-cfg')
        model_weights = config.get('model-weights')
        model_classes = config.get('model-classes')

        self.net = cv.dnn.readNetFromDarknet(model_cfg, model_weights)

        use_gpu = config.get('gpu')

        if use_gpu:
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        self.classes = None
        self.frame = None
        self.frame_json = {'detections': {}}

        self.__load_classes(model_classes)

    def __load_classes(self, model_classes):
        """
        Load class names from file
        :param model-classes: file with class names to load.
        """
        with open(model_classes, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def process(self, frame):
        """
        Main function to process the given frame
        :param frame: a frame object for detection
        """
        self.frame_json = {'detections': {}}
        self.frame = frame

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (self.input_width, self.input_height), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.__get_output_names())

        # Remove the bounding boxes with low confidence
        self.__postprocess(outs)

        # Add inference time to frame
        self.__add_inference_time()

    def __get_output_names(self):
        """
        Get the names of the output layers
        :return: names of output layers
        """
        # Get the names of all the layers in the network
        layer_names = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def __postprocess(self, outs):
        """
        Remove the bounding boxes with low confidence using non-maxima suppression
        :param frame: the frame that is being processed
        :param outs: names of output layers
        """
        frame_height = self.frame.shape[0]
        frame_width = self.frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.__add_to_json(class_ids[i], box)
            self.__draw_pred(class_ids[i], confidences[i], left, top, left + width, top + height)

    def __add_to_json(self, class_id, box):
        """
        Add the detected class to JSON
        If the class is already on JSON, add one to the counter.
        :param class_id: id of the detected object's class
        """
        class_name = self.classes[class_id]
        class_exists = self.frame_json.get('detections').get(class_name)
        box = dict(x=box[0], y=box[1], width=box[2], height=box[3])

        if class_exists is None:
            new_class = dict(count=1, boxes=[box])
            self.frame_json.get('detections').update({class_name: new_class})
        else:
            self.frame_json['detections'][class_name]['count'] += 1
            self.frame_json['detections'][class_name]['boxes'].append(box)

    def __draw_pred(self, class_id, conf, left, top, right, bottom):
        """
        Draw the predicted bounding box
        :param class_id: id of the object's detected class
        :param conf: confidence number
        :param left: left coordinates of the bounding box
        :param top: top coordinates of the bounding box
        :param right: right coordinates of the bounding box
        :param bottom: bottom coordinates of the bounding box
        """
        # Draw a bounding box.
        cv.rectangle(self.frame, (left, top), (right, bottom), (255, 178, 50), 3)

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if self.classes:
            assert (class_id < len(self.classes))
            label = '%s:%s' % (self.classes[class_id], label)

        # Display the label at the top of the bounding box
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv.rectangle(self.frame, (left, top - round(1.5 * label_size[1])),
                     (left + round(1.5 * label_size[0]), top + base_line), (255, 255, 255), cv.FILLED)
        cv.putText(self.frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    def __add_inference_time(self):
        """
        Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the
        timings for each of the layers(in layersTimes)
        """
        t, _ = self.net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(self.frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    def get_frame_json(self):
        """
        Returns a JSON object with the number of detections for each detected class in the frame.
        :return: the processed frame's detections in JSON format
        """
        return self.frame_json



