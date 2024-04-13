import os
import numpy as np
import random

import torch
from torch.utils.data import Dataset
import cv2
from pycocotools.coco import COCO

class COCODataset(Dataset):
    def __init__(self, data_dir='traffic-sign-data', 
                json_file='train.json',
                image_set='train', 
                img_size=None,
                transform=None, 
                ):
        
        self.data_dir = data_dir
        self.json_file = json_file
        self.image_set = image_set
        self.max_labels = 59
        self.img_size = img_size
        self.transform = transform
        self.coco_load, self.img_id, self.cat_id = self.load_data()
        self.cls_names = [cat['name'] for cat in self.coco_load.loadCats(self.cat_id)]

    def __len__(self):
        return len(self.img_id)
    
    def __getitem__(self, index):
        im, gt, h, w = self.load_image(index)

        return im, gt

    def load_data(self):
        coco_load = COCO(os.path.join(self.data_dir, 'annotations', self.json_file))
        # image id and category id are loaded from json file
        img_id = coco_load.getImgIds()
        print("Train images number: ", len(img_id))
        cat_id = sorted(coco_load.getCatIds())
        return coco_load, img_id, cat_id
    
    def load_label(self):
        self.cls_names[0] = 'background'
        return self.cat_id, self.cls_names
    
    def pop_img(self, index):
        id_ = self.img_id[index]
        img_name = self.coco_load.loadImgs(id_)[0]['file_name']
        img_file = os.path.join(self.data_dir, self.image_set, img_name)
        img = cv2.imread(img_file)
        return img, id_
    
    def load_image(self, index):
        # Get relevant annotations
        id_ = self.img_id[index]
        anno_ids = self.coco_load.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco_load.loadAnns(anno_ids)
        # Read image from train folder
        img_name = self.coco_load.loadImgs(id_)[0]['file_name']
        img_file = os.path.join(self.data_dir, self.image_set, img_name)
        img = cv2.imread(img_file)
        height, width, channels = img.shape
        target = []
        
        for anno in annotations:
            x1 = np.max((0, anno['bbox'][0]))
            y1 = np.max((0, anno['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, anno['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, anno['bbox'][3] - 1))))
            if anno['area'] > 0 and x2 >= x1 and y2 >= y1:
                label_ind = anno['category_id']
                cls_id = self.cat_id.index(label_ind)
                x1 /= width
                y1 /= height
                x2 /= width
                y2 /= height

                target.append([x1, y1, x2, y2, cls_id])  # [xmin, ymin, xmax, ymax, label_ind]
        # end here .

        # data augmentation
        if self.transform is not None:
            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

if __name__ == "__main__":
    from transform import Augmentation, BaseTransform

    img_size = 640
    pixel_mean = (0.406, 0.456, 0.485)  # BGR
    pixel_std = (0.225, 0.224, 0.229)   # BGR
    data_root = 'traffic-sign-data'
    transform = Augmentation(img_size, pixel_mean, pixel_std)
    transform = BaseTransform(img_size, pixel_mean, pixel_std)

    img_size = 640
    dataset = COCODataset(
                data_dir=data_root,
                img_size=img_size,
                transform=transform
                )
    print(dataset.cls_names)
    for i in range(10):
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break
        im, id_  = dataset.pop_img(i)
        im, gt, h, w = dataset.load_image(i)

        # to numpy
        image = im.permute(1, 2, 0).numpy()
        # to BGR
        image = image[..., (2, 1, 0)]
        # denormalize
        image = (image * pixel_std + pixel_mean) * 255
        # to 
        image = image.astype(np.uint8).copy()

        # draw bbox
        for box in gt:
            xmin, ymin, xmax, ymax, _ = box
            xmin *= img_size
            ymin *= img_size
            xmax *= img_size
            ymax *= img_size
            image = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), 2)
        cv2.imshow('gt', image)

