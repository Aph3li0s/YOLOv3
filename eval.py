import os
import torch
import argparse

from data_cfg.transform import BaseTransform
from cocoapi_evaluator import COCOAPIEvaluator
from utils.misc import load_weight

from models.model_cfg import build_model_config
from models.yolo import build_yolov3


parser = argparse.ArgumentParser(description='YOLOv3 Detector Evaluation')
parser.add_argument('-d', '--dataset', default='voc',
                    help='voc, coco-val, coco-test.')
parser.add_argument('--root', default='',
                    help='data root')

parser.add_argument('-v', '--version', default='yolov3',
                    help='yolo.')
parser.add_argument('--coco_test', action='store_true', default=False,
                    help='evaluate model on coco-test')
parser.add_argument('--conf_thresh', default=0.001, type=float,
                    help='得分阈值')
parser.add_argument('--nms_thresh', default=0.50, type=float,
                    help='NMS 阈值')
parser.add_argument('--topk', default=1000, type=int,
                    help='topk predicted candidates')
parser.add_argument('--weight', type=str, default=None, 
                    help='Trained state_dict file path to open')

parser.add_argument('-size', '--input_size', default=416, type=int,
                    help='input_size')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Use cuda')

args = parser.parse_args()


def coco_test(model, device, input_size, val_transform, test=False):
    data_root = os.path.join(args.root, 'COCO')
    if test:
        # test-dev
        evaluator = COCOAPIEvaluator(
            data_dir=data_root,
            img_size=input_size,
            device=device,
            testset=True,
            transform=val_transform
            )

    else:
        # eval
        evaluator = COCOAPIEvaluator(
            data_dir=data_root,
            img_size=input_size,
            device=device,
            testset=False,
            transform=val_transform
            )

    # COCO evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':
    # dataset
    if args.dataset == 'voc':
        print('eval on voc ...')
        num_classes = 20
    elif args.dataset == 'coco':
        print('eval on coco-val ...')
        num_classes = 45
    else:
        print('unknow dataset !! we only support voc, coco !!!')
        exit(0)

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cfg = build_model_config(args)

    model = build_yolov3(args, cfg, device, args.input_size, num_classes, trainable=False)
    model = load_weight(model, args.weight)
    model.to(device).eval()
    
    val_transform = BaseTransform(args.input_size)

    with torch.no_grad():
        if args.dataset == 'coco':
            if args.coco_test:
                coco_test(model, device, args.input_size, val_transform, test=True)
            else:
                coco_test(model, device, args.input_size, val_transform, test=False)
