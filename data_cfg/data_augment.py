import os
import numpy as np
import random

import torch
from torch.utils.data import Dataset
import cv2
from pycocotools.coco import COCO
from dataloader import COCODataset

class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image /= 255.
        image -= self.mean
        image /= self.std

        return image, boxes, labels
    

class Resize(object):
    def __init__(self, size=640):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels
    
class Augmentation(object):
    def __init__(self, size=640, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.mean = mean
        self.size = size
        self.std = std
        self.augment = Compose([
            RandomHue(),
            Resize(self.size),             
            Normalize(self.mean, self.std)
        ])

if __name__ == "__main__":
    from transform import BaseTransform
    img_size = 640
    pixel_mean = (0.406, 0.456, 0.485)  # BGR
    pixel_std = (0.225, 0.224, 0.229)   # BGR
    data_root = 'traffic-sign-data'
    
    transform = Augmentation(img_size, pixel_mean, pixel_std)
    # transform = BaseTransform(img_size, pixel_mean, pixel_std)
    img_size = 640
    dataset = COCODataset(
                data_dir=data_root,
                img_size=img_size,
                transform=transform
                )
    
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