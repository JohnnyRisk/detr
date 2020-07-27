import os
import cv2
import json
from tqdm import tqdm
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
from creator_helper_functions import convert_bbox_to_XYHW, convert_segmentation_to_list
import gc
from skimage.morphology import label

for phase in ['train', 'val']:
    version = 1
    json_dir = '/home/johnny/software/detr/jsons'
    json_path = os.path.join(json_dir, 'coco_aerial_{}.json'.format(phase))
    data_dir = '/data/buildings/AerialImageDataset/{}'.format(phase)
    main_categories = ["info", "licenses", "images", "annotations", "categories", "category_map"]

    info_classes = ["description", "url", "version", "year", "contributor", "date_created"]
    license_classes = ["url", "id", "name"]
    image_classes = ["id", "license", "coco_url", "flickr_url", "width", "height", "file_name", "date_captured"]
    annotations_classes = ["id", "category_id", "iscrowd", "segmentation", 'image_id', "area", "bbox"]
    category_classes = ["supercategory", "id", "name"]

    coco_json = {}

    ########### INFO ###############
    tmp_dict = {}
    tmp_dict["description"] = "INRIA Building Dataset"
    tmp_dict["url"] = "https://project.inria.fr/aerialimagelabeling/files/"
    tmp_dict["version"] = "1.0"
    tmp_dict["year"] = 2020
    tmp_dict["contributor"] = "INRIA"
    tmp_dict["date_created"] = 20200719
    coco_json["info"] = tmp_dict

    ############ LICENSES ################
    tmp_dict = {}
    tmp_dict["url"] = "unknown"
    tmp_dict["id"] = 0
    tmp_dict["name"] = "Unknown"
    coco_json["licenses"] = [tmp_dict]

    ########### IMAGES ###################
    gt_dir = os.path.join(data_dir, 'annotations')
    image_dir = os.path.join(data_dir, 'cropped')

    gts = sorted(os.listdir(gt_dir))
    images = sorted(os.listdir(image_dir))
    assert len(gts) == len(images), "number of GT's does not equal number of images"

    print('len images: {}'.format(len(images)))
    annotation_id = 0
    instance_counts = 0
    image_list = []
    annotation_list = []
    for img_id, img_fn in enumerate(images):
        base_img_path = img_fn.replace('.jpg', '')
        gt_fn = gts[img_id]
        img_path = os.path.join(image_dir, img_fn)
        gt_path = os.path.join(gt_dir, gt_fn)
        assert img_fn.replace('.jpg', '').replace('.png', '') == gts[img_id].replace('.jpg', '').replace('.png', ''), "BASE PATHS DONT MATCH"
        img_0 = cv2.imread(img_path)
        mask_0 = cv2.imread(gt_path)
        tmp_dict = {}
        tmp_dict["id"] = int(img_id)
        tmp_dict["license"] = 0
        tmp_dict["coco_url"] = 'none'
        tmp_dict["flickr_url"] = 'none'
        tmp_dict["width"] = img_0.shape[1]
        tmp_dict["height"] = img_0.shape[0]
        tmp_dict["file_name"] = img_fn
        tmp_dict["date_captured"] = 20200720
        image_list.append(tmp_dict)
        #### ANNOTATIONS
        contours, _ = cv2.findContours(mask_0[:, :, 0] // 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #lbl_0 = label(mask_0[:, :, 0]//255)
        #props = regionprops(lbl_0)
        #instance_counts += len(props)
        for contour in contours:
            if contour.squeeze(axis=1).shape[0] >= 5:
                ann_dict = {}
                ann_dict["id"] = annotation_id
                ann_dict["category_id"] = 0
                ann_dict["iscrowd"] = 0
                ann_dict["segmentation"] = convert_segmentation_to_list(contour)
                ann_dict["image_id"] = img_id
                ann_dict["bbox"] = convert_bbox_to_XYHW(contour)
                ann_dict["area"] = float(ann_dict["bbox"][2]*ann_dict["bbox"][3])
                annotation_id += 1
                annotation_list.append(ann_dict)

        if img_id % 100 == 0:
            gc.collect()
            print('Processed: {} / {} {} images'.format(img_id, len(images), phase))

    coco_json['images'] = image_list
    coco_json['annotations'] = annotation_list

    ########### CATEGORIES ################

    tmp_dict = {}
    cat_list = []
    tmp_dict["supercategory"] = "building"
    tmp_dict["id"] = 0
    tmp_dict["name"] = "building"
    coco_json["categories"] = [tmp_dict]
    coco_json["category_map"] = {"building": 0}
    print('{} instances in the {} set'.format(annotation_id, phase))

    with open(json_path, 'w') as f:
        json.dump(coco_json, f)
