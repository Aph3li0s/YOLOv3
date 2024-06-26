import json
import tempfile
import torch
import numpy as np
from pycocotools.cocoeval import COCOeval
from data_cfg.dataloader import COCODataset


class COCOAPIEvaluator():
    def __init__(self, data_dir, img_size, device, testset=False, transform=None):
        self.img_size = img_size
        self.transform = transform
        self.device = device
        self.map = -1.

        self.testset = testset
        if self.testset:
            json_file='test.json'
            image_set = 'test'
            
        else:
            json_file='valid.json'
            image_set='valid'

        self.dataset = COCODataset(
            data_dir=data_dir,
            img_size=img_size,
            json_file=json_file,
            transform=None,
            image_set=image_set)


    def evaluate(self, model):
        model.eval()
        ids = []
        data_dict = []
        num_images = len(self.dataset)
        print('total number of images: %d' % (num_images))

        # start testing
        for index in range(num_images): # all the data in val2017
            if index % 200 == 0:
                print('[Eval: %d / %d]'%(index, num_images))

            img, id_ = self.dataset.pop_img(index)  # load a batch
            if self.transform is not None:
                x = torch.from_numpy(self.transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
                x = x.unsqueeze(0).to(self.device)
            scale = np.array([[img.shape[1], img.shape[0],
                            img.shape[1], img.shape[0]]])
            
            id_ = int(id_)
            ids.append(id_)
            with torch.no_grad():
                outputs = model(x)
                bboxes, scores, labels = outputs
                bboxes *= scale
            for i, box in enumerate(bboxes):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                label = self.dataset.cat_id[int(labels[i])]
                
                bbox = [x1, y1, x2 - x1, y2 - y1]
                score = float(scores[i]) # object score * class score
                A = {"image_id": id_, "category_id": label, "bbox": bbox,
                    "score": score} # COCO json format
                data_dict.append(A)

        annType = ['segm', 'bbox', 'keypoints']

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            print('evaluating ......')
            cocoGt = self.dataset.coco_load
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            if self.testset:
                json.dump(data_dict, open('predict_coco.json', 'w'))
                cocoDt = cocoGt.loadRes('predict_coco.json')
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, 'w'))
                cocoDt = cocoGt.loadRes(tmp)
            cocoEval = COCOeval(self.dataset.coco_load, cocoDt, annType[1])
            cocoEval.params.imgIds = ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            ap50_95, ap50 = cocoEval.stats[0], cocoEval.stats[1]
            print('ap50_95 : ', ap50_95)
            print('ap50 : ', ap50)
            self.map = ap50_95
            self.ap50_95 = ap50_95
            self.ap50 = ap50

            return ap50_95, ap50
        else:
            return 0, 0

