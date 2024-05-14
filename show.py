import argparse
import torch
import numpy as np
import cv2
import os
import time

from utils.misc import load_weight
from data_cfg.dataloader import COCODataset
from data_cfg.transform import BaseTransform

from models.model_cfg import build_model_config
from models.yolo import build_yolov3


parser = argparse.ArgumentParser(description='YOLOv3 Show')
parser.add_argument('--root', default='',
                    help='data root')
parser.add_argument('-size', '--input_size', default=640, type=int,
                    help='input size of image')
parser.add_argument('-d', '--dataset', default='coco',
                    help='voc, coco-val.')
parser.add_argument('-v', '--version', default='yolov3',
                    help='yolo')
parser.add_argument('--weight', default=None,
                    type=str, help='weight file path')
parser.add_argument('--conf_thresh', default=0.1, type=float,
                    help='')
parser.add_argument('--nms_thresh', default=0.50, type=float,
                    help='NMS')
parser.add_argument('--topk', default=100, type=int,
                    help='topk predicted candidates')
                    
parser.add_argument('-vs', '--visual_threshold', default=0.20, type=float,
                    help='')
parser.add_argument('--cuda', action='store_true', default=True, 
                    help='use cuda.')
parser.add_argument('--img', default='test.jpg', type=str,
                    help='image path')

args = parser.parse_args()

def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img

def visualize(img, 
            bboxes, 
            scores, 
            labels, 
            vis_thresh, 
            class_colors, 
            class_names, 
            class_indexs=None):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        # if scores[i] > vis_thresh:
        cls_id = int(labels[i])
        cls_color = class_colors[cls_id]
        cls_id = class_indexs[cls_id]
        if len(class_names) > 1:
            mess = '%s: %.2f' % (class_names[cls_id], scores[i])
        else:
            cls_color = [255, 0, 0]
            mess = None
        img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)

    return img

def test(args, model, device, image_dir, transform, class_colors=None, class_names=None, class_indexs=None):
    img = cv2.imread(image_dir)
    h, w, _ = img.shape

    x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
    x = x.unsqueeze(0).to(device)

    t0 = time.time()

    bboxes, scores, labels = model(x)
    # print("detection time used ", time.time() - t0, "s")
    scale = np.array([[w, h, w, h]])
    bboxes *= scale

    img_processed = visualize(
        img=img,
        bboxes=bboxes,
        scores=scores,
        labels=labels,
        vis_thresh=args.visual_threshold,
        class_colors=class_colors,
        class_names=class_names,
        class_indexs=class_indexs
        )
    cv2.imwrite('det.jpg', img_processed)
    # cv2.imshow('detection', img_processed)     
    # cv2.waitKey(0)
    
if __name__ == '__main__':
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    input_size = args.input_size

    if args.dataset == 'coco':
        data_root = os.path.join(args.root, 'traffic-sign_5k')
        dataset = COCODataset(
                    data_dir=data_root,
                    json_file='test.json',
                    image_set='test',
                    img_size=input_size)
        class_names = dataset.cls_names
        class_indexs = dataset.cat_id
        num_classes = len(class_indexs) -1 
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                    np.random.randint(255),
                    np.random.randint(255)) for _ in range(num_classes)]
    t0 = time.time()
    cfg = build_model_config(args)

    model = build_yolov3(args, cfg, device, input_size, num_classes, trainable=False)
    
    model = load_weight(model, args.weight)
    model.to(device).eval()

    val_transform = BaseTransform(input_size)
    test(args=args,
        model=model, 
        device=device, 
        image_dir=args.img,
        transform=val_transform,
        class_colors=class_colors,
        class_names=class_names,
        class_indexs=class_indexs,
        )
    print("Time taken: ", time.time() - t0)