import os
import pickle
import xml.etree.ElementTree as ET
import argparse
from PIL import Image
from tqdm import tqdm

from lib.utils.utils import get_classes
from lib.utils.utils_map import get_coco_map, get_map
from yolo import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arguement')
    parser.add_argument('--train_mode', type=int, default=0)
    parser.add_argument('--flag', type=int, default=0)
    parser.add_argument('--dataset', type=int, default=0)
    args = parser.parse_args()
    #0 gaze estimation and object detection
    #1 object detection
    train_mode=args.train_mode
    #wUOC or map
    #0 get map
    #1 get wUOC
    flag=args.flag
    # ------------------------------------------------------------------------------------------------------------------#
    #   map_mode is used to specify the content calculated when the file is running
    # -------------------------------------------------------------------------------------------------------------------#
    map_mode = 0
    # -------------------------------------------------------#
    # The classes_path here is used to specify the category of VOC_map that needs to be measured
    # Generally, it is consistent with the classes_path used for training and prediction.
    # -------------------------------------------------------#
    classes_path = 'data/anchors/voc_classes.txt'
    # -------------------------------------------------------#
    # MINOVERLAP is used to specify the mAP0.x you want to obtain
    # For example, to calculate mAP0.75, you can set MINOVERLAP = 0.75.
    # -------------------------------------------------------#
    MINOVERLAP = 0.5
    # -------------------------------------------------------#
    #The dataset variable is used to specify the dataset that needs to be predicted(GOO-synth or GOO-real)
    #0 stands for GOO-synth
    #1 stands for GOO-real
    dataset=args.dataset
    #--------------------------------------------------------#
    #   map_vis is used to specify whether to enable the visualization of VOC_map calculations
    # -------------------------------------------------------#
    map_vis = False
    # -------------------------------------------------------#
    # Point to the folder where the VOC data set is located
    # By default, it points to the VOC data set in the root directory
    # -------------------------------------------------------#
    VOCdevkit_path = 'test_data/VOCdevkit'
    # -------------------------------------------------------#
    #   The output folder, the default is map_out
    # -------------------------------------------------------#
    map_out_path = './data/map_out'

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if dataset==1:
        if not os.path.exists(os.path.join(map_out_path, 'ground-truth-real')):
            os.makedirs(os.path.join(map_out_path, 'ground-truth-real'))
        if not os.path.exists(os.path.join(map_out_path, 'detection-real-results')):
            os.makedirs(os.path.join(map_out_path, 'detection-real-results'))
        if not os.path.exists(os.path.join(map_out_path, 'images-real-optional')):
            os.makedirs(os.path.join(map_out_path, 'images-real-optional'))
    if dataset==0:
        if not os.path.exists(os.path.join(map_out_path, 'ground-truth-synth')):
            os.makedirs(os.path.join(map_out_path, 'ground-truth-synth'))
        if not os.path.exists(os.path.join(map_out_path, 'detection-synth-results')):
            os.makedirs(os.path.join(map_out_path, 'detection-synth-results'))
        if not os.path.exists(os.path.join(map_out_path, 'images-synth-optional')):
            os.makedirs(os.path.join(map_out_path, 'images-synth-optional'))
    class_names, _ = get_classes(classes_path)

    if (map_mode == 0 or map_mode == 1) and dataset==1:
        print("Load model.")
        yolo = YOLO(confidence=0.001, nms_iou=0.5)
        print("Load model done.")
        image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()
        print("Get GOO-real predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-real-optional/" + image_id + ".jpg"))
            yolo.get_map_txt(dataset,train_mode,image_id, image, class_names, map_out_path)
        print("Get GOO-real predict result done.")
    if (map_mode == 0 or map_mode == 1) and dataset==0:
        print("Load model.")
        yolo = YOLO(confidence=0.001, nms_iou=0.5)
        print("Load model done.")
        images_path = 'test_data/data_proc/images1/'
        image_ids = open("test_data/data_proc/test.txt").read().strip().split()
        print("Get GOO-synth predict result.")
        for image_id in tqdm(image_ids):
            image_path = images_path + image_id + ".png"
            image = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-synth-optional/" + image_id + ".png"))
            yolo.get_map_txt(dataset,train_mode,image_id, image, class_names, map_out_path)
        print("Get GOO-synth predict result done.")
    if (map_mode == 0 or map_mode == 2) and dataset==1:
        image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()
        print("Get GOO-real ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth-real/" + image_id + ".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/" + image_id + ".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') != None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text
                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get GOO-real ground truth result done.")
    if (map_mode == 0 or map_mode == 2) and dataset == 0:
        images_path =os.getcwd()+  'test_data/data_proc/images1/'
        data_path =os.getcwd()+  'test_data/data_proc/data1/'
        out_path = map_out_path + '/ground-truth-synth/'
        class_list, _ = get_classes(classes_path)
        file_list = sorted(os.listdir(images_path))
        index = 0
        print("Starting get GOO-synth ground truth...")
        for item in file_list:
            with open(data_path + str(index) + '.pickle', 'rb') as f:
                pkl = pickle.load(f)
                with open(out_path + str(index) + '.txt', 'w+') as f1:
                    gt_box = pkl['ann']['bboxes']
                    gt_label = pkl['ann']['labels']
                    for box, label in zip(gt_box, gt_label):
                        f1.write(class_names[label] + ' ')
                        for x in box:
                            f1.write(str(int(x)) + ' ')
                        f1.write('\n')
            index = index + 1
        print("Get GOO-synth ground truth done!")
    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP,dataset,flag,True,path=map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(dataset,class_names=class_names, path=map_out_path)
        print("Get map done.")
