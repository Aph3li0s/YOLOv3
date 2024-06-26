# yolov3 config


yolov3_config = {
    'yolov3': {
        # model
        'backbone': 'darknet53',
        'pretrained': True,
        'stride': [8, 16, 32],  # P5
        'head_dim': [256, 512, 1024],
        # anchor size
        'anchor_size': {
            'coco': [[12.48, 19.2],   [31.36, 46.4],    [46.4, 113.92],
                    [97.28, 55.04],  [133.12, 127.36], [79.04, 224.],
                    [301.12, 150.4], [172.16, 285.76], [348.16, 341.12]]
            },
        # matcher
        'ignore_thresh': 0.5,
        },
}
def build_model_config(args):
    return yolov3_config[args.version]