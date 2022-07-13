import pandas 
import os
import os.path as osp
import random
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import numpy as np

class COCOeval_wmAP(COCOeval):
    def __init__(self, gt_coco, res_coco, iouType='segm', num_cls=107, alpha=10):
        super(COCOeval_wmAP, self).__init__(gt_coco, res_coco, iouType)
        self.num_cls = num_cls
        self.weights = np.array([1 if i!=106 else alpha for i in range(num_cls)])


    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                s[s==-1] =0
                mean_s = np.mean(np.average(s, weights=self.weights, axis=-2))
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

def convert_coco(df):
    "Convert dataframe to COCO json format"
    coco_dict = {"images": [], "annotations": [], "categories": []}
    for i, row in df.iterrows():
        image_id = i
        image_name = row["image_name"]
        image_width = 200
        image_height = 200
        coco_dict["images"].append({"id": image_id, "file_name": image_name, "width": image_width, "height": image_height})
        category_id = row["class_id"]
        category_name = str(category_id)
        area = (row["x_max"] - row["x_min"]) * (row["y_max"] - row["y_min"])
        coco_dict["categories"].append({"id": category_id, "name": category_name})
        coco_dict["annotations"].append({"id": i, "score": row["confidence_score"], 
                                         "image_id": image_id, "category_id": category_id, "area": area,
                                         "bbox": [row["x_min"], row["y_min"], row["x_max"], row["y_max"]], "iscrowd": 0})
    return coco_dict

def wmAP(gt_df, pred_df):
    gt_coco = convert_coco(gt_df)
    pred_coco = convert_coco(pred_df)
    json.dump(gt_coco, open("gt.json", "w"))
    json.dump(pred_coco, open("pred.json", "w"))
    cocoGt = COCO("gt.json")
    cocoPt=COCO("pred.json")
    cocoEval = COCOeval_wmAP(cocoGt, cocoPt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats[0]

if __name__ == "__main__":
    pred_df = pandas.read_csv("gt.csv")
    gt_df = pandas.read_csv("gt.csv")
    wmap = wmAP(gt_df, pred_df)