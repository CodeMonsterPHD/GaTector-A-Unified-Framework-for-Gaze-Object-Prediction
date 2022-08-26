import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont
from torchvision import transforms
from lib.nets.gatector import GaTectorBody
from lib.utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image)
from lib.utils.utils_bbox import DecodeBox



class YOLO(object):
    _defaults = {
        #Please modify it to the training mode
        "train_mode":0,
        #Please change the model path
        "model_path": '/data1/jinyang/GaTector/logsSync/ep100-loss15.911-val_loss43.973.pth',
        "classes_path": 'data/anchors/voc_classes.txt',
        # ---------------------------------------------------------------------#
        # anchors_path represents the txt file corresponding to the a priori box, which is generally not modified.
        # anchors_mask is used to help the code find the corresponding a priori box, generally not modified.
        # ---------------------------------------------------------------------#
        "anchors_path": 'data/anchors/yolo_anchors.txt',
        "anchors_mask": [[0, 1, 2]],

        "anchors_path_3": 'data/anchors/yolo_anchors_3.txt',
        "anchors_mask_3": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        # ---------------------------------------------------------------------#
        #   The size of the input image must be a multiple of 32.
        # ---------------------------------------------------------------------#
        "input_shape": [224, 224],
        # ---------------------------------------------------------------------#
        #   Only prediction boxes with a score greater than the confidence level will be retained
        # ---------------------------------------------------------------------#
        "confidence": 0.5,
        # ---------------------------------------------------------------------#
        #   The size of nms_iou used for non-maximum suppression
        # ---------------------------------------------------------------------#
        "nms_iou": 0.3,
        # ---------------------------------------------------------------------#
        #   This variable is used to control whether to use letterbox_image to resize the input image without distortion
        # ---------------------------------------------------------------------#
        "letterbox_image": False,

        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   Initialize YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # ---------------------------------------------------#
        #   Get the number of types and a priori boxes
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        if self.train_mode==0:
            self.anchors, self.num_anchors = get_anchors(self.anchors_path)
            self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                       self.anchors_mask)
        if self.train_mode == 1:
            self.anchors, self.num_anchors = get_anchors(self.anchors_path_3)
            self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                       self.anchors_mask_3)

        # ---------------------------------------------------#
        #   Set different colors for the picture frame
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    # ---------------------------------------------------#
    #   Generative model
    # ---------------------------------------------------#
    def generate(self):
        if self.train_mode==0:
            self.net = GaTectorBody(self.anchors_mask, self.num_classes, self.train_mode)
        if self.train_mode == 1:
            self.net = GaTectorBody(self.anchors_mask_3, self.num_classes, self.train_mode)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.net = nn.DataParallel(self.net)
        # self.net = self.net.cuda()
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, image,train_mode):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        self.transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        image_data = torch.Tensor(image_data)
        image_data = self.transform(image_data)
        image_data = np.array(image_data, dtype=np.float32)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                face = images
                head = np.ones([1, 1, 224, 224])
                head = torch.from_numpy(head).type(torch.FloatTensor).cuda()
            outputs = self.net(images, head, face,train_mode)
            outputs = outputs[1:]
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
        # ---------------------------------------------------------#
        #   Set font and border thickness
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        # ---------------------------------------------------------#
        #   Image drawing
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   Input images into the network for prediction
            # ---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            # ---------------------------------------------------------#
            #   Stack the prediction boxes, and then perform non-maximum suppression
            # ---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------------#
                #   Input images into the network for prediction
                # ---------------------------------------------------------#
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                # ---------------------------------------------------------#
                #   Stack the prediction boxes, and then perform non-maximum suppression
                # ---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                             image_shape, self.letterbox_image,
                                                             conf_thres=self.confidence, nms_thres=self.nms_iou)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, dataset,train_mode,image_id, image, class_names, map_out_path):
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        # Convert the image to an RGB image here to prevent the grayscale image from reporting errors during prediction.
        # The code only supports the prediction of RGB images, all other types of images will be converted into RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        # Add gray bars to the image to achieve undistorted resize
        # You can also directly resize for identification
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   Add the batch_size dimension
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        self.transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        image_data = torch.Tensor(image_data)
        image_data = self.transform(image_data)
        image_data = np.array(image_data, dtype=np.float32)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                # For mAP, face and head are useless
                face = images
                head = np.ones([1, 1, 224, 224])
                head = torch.from_numpy(head).type(torch.FloatTensor).cuda()
            # ---------------------------------------------------------#
            #   Input images into the network for prediction
            # ---------------------------------------------------------#
            outputs = self.net(images, head, face,train_mode)
            outputs = outputs[1:]
            outputs = self.bbox_util.decode_box(outputs)
            # ---------------------------------------------------------#
            #   Stack the prediction boxes, and then perform non-maximum suppression
            # ---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue
            if dataset == 1:
                f = open(os.path.join(map_out_path, "detection-real-results/" + image_id + ".txt"), "a+")
                f.write("%s %s %s %s %s %s\n" % (
                    predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
            if dataset == 0:
                f = open(os.path.join(map_out_path, "detection-synth-results/" + image_id + ".txt"), "a+")
                f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
        f.close()
        return
