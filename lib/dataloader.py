from random import sample, shuffle
import pickle
import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from lib.utils.utils import cvtColor, preprocess_input
from lib import gaze_imutils
import torch
import torchvision.transforms.functional as TF


class GaTectorDataset(Dataset):
    def __init__(self, root_dir, mat_file, input_shape, num_classes, train_mode,train):
        super(GaTectorDataset, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.train_mode=train_mode

        # GOO pickle
        self.output_size = 64
        self.input_size = 224
        self.root_dir = root_dir
        self.mat_file = mat_file
        with open(mat_file, 'rb') as f:
            self.data = pickle.load(f)
            self.image_num = len(self.data)
        self.transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                             ])

    def __len__(self):
        return self.image_num

    def __getitem__(self, index):
        index = index % self.image_num

        # GOO pickle
        data = self.data[index]
        image_path = data['filename']
        image_path = os.path.join(self.root_dir, image_path)
        image_path = image_path.replace('\\', '/')
        gt_box_idx = data['gazeIdx']
        # Goo gt_box
        if self.train_mode==0:
            gt_bboxes = np.copy(data['ann']['bboxes'])
            gt_labels = np.copy(data['ann']['labels'])
        if self.train_mode==1:
            gt_bboxes = np.copy(data['ann']['bboxes']) / [640, 480, 640, 480] * [1920, 1080, 1920, 1080]
            gt_labels = np.copy(data['ann']['labels'])

        gt_labels = gt_labels[..., np.newaxis]
        bbox = np.append(gt_bboxes, gt_labels, axis=1)
        box = bbox.astype(np.int32)

        gaze_gt_box = box[gt_box_idx]
        gaze_gt_box = gaze_gt_box[np.newaxis, :]

        # GOO
        eye = [float(data['hx']) / 640, float(data['hy']) / 480]
        gaze = [float(data['gaze_cx']) / 640, float(data['gaze_cy']) / 480]
        img = Image.open(image_path)
        img = img.convert('RGB')
        width, height = img.size
        gaze_x, gaze_y = gaze
        eye_x, eye_y = eye

        k = 0.1
        x_min = (eye_x - 0.15) * width
        y_min = (eye_y - 0.15) * height
        x_max = (eye_x + 0.15) * width
        y_max = (eye_y + 0.15) * height
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if x_max < 0:
            x_max = 0
        if y_max < 0:
            y_max = 0
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += k * abs(y_max - y_min)
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

        if self.train:
            # data augmentation
            # Jitter (expansion-only) bounding box size
            if np.random.random_sample() <= 0.5:
                k = np.random.random_sample() * 0.2
                x_min -= k * abs(x_max - x_min)
                y_min -= k * abs(y_max - y_min)
                x_max += k * abs(x_max - x_min)
                y_max += k * abs(y_max - y_min)

            # Random Crop
            if np.random.random_sample() <= 0.5:
                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                crop_x_min = np.min([gaze_x * width, x_min, x_max])
                crop_y_min = np.min([gaze_y * height, y_min, y_max])
                crop_x_max = np.max([gaze_x * width, x_min, x_max])
                crop_y_max = np.max([gaze_y * height, y_min, y_max])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)

                # Crop it
                img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)

                # Record the crop's (x, y) offset
                offset_x, offset_y = crop_x_min, crop_y_min

                # convert coordinates into the cropped frame
                x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y
                # if gaze_inside:
                gaze_x, gaze_y = (gaze_x * width - offset_x) / float(crop_width), \
                                 (gaze_y * height - offset_y) / float(crop_height)

                width, height = crop_width, crop_height

                box[:, [0, 2]] = box[:, [0, 2]] - crop_x_min
                box[:, [1, 3]] = box[:, [1, 3]] - crop_y_min

                # operate gt_box
                gaze_gt_box[:, [0, 2]] = gaze_gt_box[:, [0, 2]] - crop_x_min
                gaze_gt_box[:, [1, 3]] = gaze_gt_box[:, [1, 3]] - crop_y_min

            # Random flip
            if np.random.random_sample() <= 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                x_max_2 = width - x_min
                x_min_2 = width - x_max
                x_max = x_max_2
                x_min = x_min_2
                gaze_x = 1 - gaze_x
                box[:, [0, 2]] = width - box[:, [2, 0]]

            # Random color change
            if np.random.random_sample() <= 0.5:
                img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

            # Random color change
            if np.random.random_sample() <= 0.5:
                img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        head_channel = gaze_imutils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                          resolution=self.input_size, coordconv=False).unsqueeze(0)

        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        face = face.resize((self.input_shape), Image.BICUBIC)
        face = np.transpose(preprocess_input(np.array(face, dtype=np.float32)), (2, 0, 1))
        face = torch.Tensor(face)
        face = self.transform(face)
        img = img.resize((self.input_shape), Image.BICUBIC)
        img = np.transpose(preprocess_input(np.array(img, dtype=np.float32)), (2, 0, 1))
        img = torch.Tensor(img)
        img = self.transform(img)

        # Bbox deal
        box[:, [0, 2]] = box[:, [0, 2]] * self.input_size / width
        box[:, [1, 3]] = box[:, [1, 3]] * self.input_size / height

        # operate_gt_box
        gaze_gt_box[:, [0, 2]] = gaze_gt_box[:, [0, 2]] * self.input_size / width
        gaze_gt_box[:, [1, 3]] = gaze_gt_box[:, [1, 3]] * self.input_size / height

        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > self.input_size] = self.input_size
        box[:, 3][box[:, 3] > self.input_size] = self.input_size
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]

        box = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

        # generate the heatmap used for deconv prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
        gaze_heatmap = gaze_imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                   3,
                                                   type='Gaussian')
        face = np.array(face, dtype=np.float32)
        img = np.array(img, dtype=np.float32)
        head_channel = np.array(head_channel, dtype=np.float32)
        gaze_heatmap = np.array(gaze_heatmap, dtype=np.float32)

        return img, box, face, head_channel, gaze_heatmap, eye, gaze, gaze_gt_box

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

# DataLoader_collate_fn
def gatector_dataset_collate(batch):
    images = []
    bboxes = []
    face = []
    head_channel = []
    gaze_heatmap = []
    eye = []
    gaze = []
    gt_boxes = []
    for img, box, face_, head, heatmap, eyes, gazes, gt_box in batch:
        images.append(img)
        bboxes.append(box)
        face.append(face_)
        head_channel.append(head)
        gaze_heatmap.append(heatmap)
        eye.append(eyes)
        gaze.append(gazes)
        gt_boxes.append(gt_box)
    images = np.array(images)
    face = np.array(face)
    head_channel = np.array(head_channel)
    gaze_heatmap = np.array(gaze_heatmap)
    eye = np.array(eye)
    gaze = np.array(gaze)
    gt_boxes = np.array(gt_boxes)
    return images, bboxes, face, head_channel, gaze_heatmap, eye, gaze, gt_boxes
