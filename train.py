from __future__ import division

import os
import random
import argparse
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

from data_cfg.dataloader import COCODataset
from data_cfg.transform import Augmentation, BaseTransform

from utils.misc import detection_collate
from utils.com_paras_flops import FLOPs_and_Params
from cocoapi_evaluator import COCOAPIEvaluator

from models.model_cfg import build_model_config
from models.yolo import build_yolov3
from models.matcher import gt_creator

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv3 Detection')

    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--eval_epoch', type=int,
                            default=5, help='interval between evaluations')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    
    parser.add_argument('-v', '--version', default='yolov3',
                        help='build yolo')
    parser.add_argument('--conf_thresh', default=0.001, type=float,
                        help='Confidence threshold')
    parser.add_argument('--nms_thresh', default=0.50, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int,
                        help='topk predicted candidates')

    parser.add_argument('-bs', '--batch_size', default=8, type=int, 
                        help='Batch size for training')
    parser.add_argument('-accu', '--accumulate', default=8, type=int, 
                        help='gradient accumulate.')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=1,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='keep training')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')                  
    parser.add_argument('--max_epoch', type=int, default=250,
                        help='The upper bound of warm-up')
    parser.add_argument('--lr_epoch', nargs='+', default=[150, 200], type=int,
                        help='lr epoch to decay')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco')
    parser.add_argument('--root', default='',
                        help='data root')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)
    
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.multi_scale:
        print('use the multi-scale trick ...')
        train_size = 640
        val_size = 416
    else:
        train_size = 416
        val_size = 416
        
    cfg = build_model_config(args)

    dataset, num_classes, evaluator = build_dataset(args, device, train_size, val_size)
    dataloader = build_dataloader(args, dataset)
    
    model = build_yolov3(args, cfg, device, train_size, num_classes, trainable=True)
    model.to(device).train()

    # compute FLOPs and Params
    model_copy = deepcopy(model)
    model_copy.trainable = False
    model_copy.eval()
    FLOPs_and_Params(model=model_copy, 
                        img_size=val_size, 
                        device=device)
    del model_copy

    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/coco/', args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)
    
    # keep training
    if args.resume is not None:
        print('keep training model: %s' % (args.resume))
        model.load_state_dict(torch.load(args.resume, map_location=device))
        
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), 
                            lr=args.lr, 
                            momentum=args.momentum,
                            weight_decay=args.weight_decay
                            )

    max_epoch = args.max_epoch    
    lr_epoch = args.lr_epoch
    epoch_size = len(dataloader)  

    best_map = -1.
    t0 = time.time()
    for epoch in range(args.start_epoch, max_epoch):
        if epoch in lr_epoch:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)


        for iter_i, (images, targets) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    nw = args.wp_epoch*epoch_size
                    tmp_lr = base_lr * pow((ni)*1. / (nw), 4)
                    set_lr(optimizer, tmp_lr)

                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)

            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                train_size = random.randint(10, 19) * 32
                model.set_grid(train_size)
            if args.multi_scale:
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)

            targets = [label.tolist() for label in targets]
            targets = gt_creator(
                input_size=train_size, 
                strides=cfg['stride'], 
                label_lists=targets, 
                anchor_size=cfg['anchor_size'][args.dataset],
                ignore_thresh=cfg['ignore_thresh']
                )

            # to device
            images = images.to(device)
            targets = targets.to(device)

            conf_loss, cls_loss, bbox_loss, total_loss = model(images, targets=targets)
                        
            total_loss /= args.accumulate
            total_loss.backward()        
            
            if ni % args.accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            if iter_i % 50 == 0:
                if args.tfboard:
                    # viz loss
                    writer.add_scalar('obj loss', conf_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('cls loss', cls_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('box loss', bbox_loss.item(), iter_i + epoch * epoch_size)
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                        % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                            conf_loss.item(), 
                            cls_loss.item(), 
                            bbox_loss.item(), 
                            total_loss.item(), 
                            train_size, t1-t0),
                        flush=True)

                t0 = time.time()

        # evaluation
        if epoch  % args.eval_epoch == 0 or (epoch + 1) == max_epoch:
            model.trainable = False
            model.set_grid(val_size)
            model.eval()

            # evaluate
            evaluator.evaluate(model)

            # convert to training mode.
            model.trainable = True
            model.set_grid(train_size)
            model.train()

            cur_map = evaluator.map
            if cur_map > best_map:
                # update best-map
                best_map = cur_map
                # save model
                print('Saving state, epoch:', epoch + 1)
                weight_name = '{}_epoch_{}_{:.1f}.pth'.format(args.version, epoch + 1, best_map*100)
                checkpoint_path = os.path.join(path_to_save, weight_name)
                torch.save(model.state_dict(), checkpoint_path)                      


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def build_dataset(args, device, train_size, val_size):
    pixel_mean = (0.406, 0.456, 0.485)  # BGR
    pixel_std = (0.225, 0.224, 0.229)   # BGR
    train_transform = Augmentation(train_size, pixel_mean, pixel_std)
    val_transform = BaseTransform(val_size, pixel_mean, pixel_std)


    data_root = os.path.join(args.root, 'traffic-sign_5k')
    num_classes = 45
    dataset = COCODataset(
        data_dir=data_root,
        img_size=train_size,
        transform=train_transform
        )

    evaluator = COCOAPIEvaluator(
        data_dir=data_root,
        img_size=val_size,
        device=device,
        transform=val_transform
        )

    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")


    return dataset, num_classes, evaluator


def build_dataloader(args, dataset):
    dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    collate_fn=detection_collate,
                    num_workers=args.num_workers,
                    pin_memory=True
                    )
    
    return dataloader


if __name__ == '__main__':
    train()
