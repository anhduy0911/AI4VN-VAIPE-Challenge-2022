import pandas 
import os
import os.path as osp
import random
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import numpy as np

class COCOeval_wmAP(COCOeval):
    def __init__(self, gt_coco, res_coco, iouType='segm', num_cls=107, alpha=100):
        super(COCOeval_wmAP, self).__init__(gt_coco, res_coco, iouType)
        self.num_cls = num_cls
        # self.weights = np.array([1 if i!=106 else alpha for i in range(num_cls)])
        self.alpha = alpha

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
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
                n_cls = s.shape[-2]
                weights = np.array([1 if i!=106 else self.alpha for i in range(n_cls)])
                mean_s = np.mean(np.average(s, weights=weights, axis=-2))
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
    seen_labs = []
    seen_img_names = []
    for i, row in df.iterrows():
        image_id = row["image_name"].split('.')[0]
        image_name = row["image_name"]
        image_width = row["image_width"]
        image_height = row["image_height"]
        if image_id not in seen_img_names:
            coco_dict["images"].append({"id": image_id, "file_name": image_name, "width": image_width, "height": image_height})
            seen_img_names.append(image_id)
        category_id = row["class_id"]
        category_name = str(category_id)
        area = (row["x_max"] - row["x_min"]) * (row["y_max"] - row["y_min"])
        if category_id not in seen_labs:
            coco_dict["categories"].append({"id": category_id, "name": category_name})
            seen_labs.append(category_id)
        coco_dict["annotations"].append({"id": i, "score": row["confidence_score"], 
                                         "image_id": image_id, "category_id": category_id, "area": area,
                                         "bbox": [row["x_min"], row["y_min"], row["x_max"] - row["x_min"], row["y_max"] - row["y_min"]], "iscrowd": 0})
    
    # coco_dict["categories"] = set(coco_dict["categories"])
    # for i in range(107):
    #     coco_dict["categories"].append({"id": i, "name": str(i)})

    return coco_dict

def convert_coco_predict(df):
    result = []
    for i, row in df.iterrows():
        row_i = {}
        row_i['image_id'] = row['image_name'].split('.')[0]
        row_i['category_id'] = row['class_id']
        row_i['bbox'] = [row["x_min"], row["y_min"], row["x_max"] - row["x_min"], row["y_max"] - row["y_min"]]
        row_i['score'] = row['confidence_score']
        result.append(row_i)
    
    return result

def wmAP(gt_df, pred_df, output_dir):
    gt_coco = convert_coco(gt_df)
    pred_coco = convert_coco_predict(pred_df)
    # pred_coco = convert_coco(pred_df, truth=False, img_meta_df=img_meta)
    json.dump(gt_coco, open(os.path.join(output_dir,"gt.json"), "w"))
    json.dump(pred_coco, open(os.path.join(output_dir,"pred.json"), "w"))
    cocoGt = COCO(os.path.join(output_dir,"gt.json"))
    cocoPt = cocoGt.loadRes(os.path.join(output_dir,"pred.json"))
    cocoEval = COCOeval_wmAP(cocoGt, cocoPt, iouType='bbox', alpha=3)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats[:6]

def do_evaluation(gtruth_path, predict_path, output_dir):
    '''
    Do validation for the data
    '''
    def validate_data():
        '''
        Do validation for the dataset
        ''' 
        gt_df = pandas.read_csv(gtruth_path)
        pred_df = pandas.read_csv(predict_path)
        return gt_df, pred_df

    print('Let load the data')
    gt_df, pred_df = validate_data()
    print('Load finish')
    metrics = ['wmAP', 'wmAP50','wmAP75', 'wmAPs', 'wmAPm', 'wmAPl']
    result = wmAP(gt_df, pred_df, output_dir)
    print('calculate wmap finish')
    result_dict = {}
    for i, k in enumerate(metrics):
        result_dict[k] = result[i]
    
    return result_dict

if __name__ == "__main__":
    # pred_df = pandas.read_csv("groundtruth.csv")
    # gt_df = pandas.read_csv("groundtruth.csv")
    # wmap = wmAP(gt_df, pred_df)
    # print(wmap)
    print(do_evaluation('groundtruth.csv', 'predictions.csv'))