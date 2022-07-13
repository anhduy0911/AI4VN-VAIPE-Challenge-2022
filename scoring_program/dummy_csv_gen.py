import pandas  as pd
import os
import os.path as osp
import random

##Hyperparameters controlling prediction quality
alpha = 0.95
predicted_acc = 0.8
extra_box_rate = 0.1

gt = {"image_name": [], "class_id": [], "confidence_score": [], "x_min": [], "y_min": [], "x_max": [], "y_max": []}
for image_id in range(1000):
    image_name = osp.join("data", "public_test", "image_%d.jpg" % image_id)
    for box_id in range(random.randint(1, 3)):
        if random.random() < alpha:
            class_id = random.randint(0, 105)
        else:
            class_id = 106
        gt["image_name"].append(image_name)
        gt["class_id"].append(class_id)
        gt["confidence_score"].append(1.0)
        gt["x_min"].append(int(random.uniform(0, 200)))
        gt["x_max"].append(int(random.uniform(gt["x_min"][-1], 200)))
        gt["y_min"].append(int(random.uniform(0, 200)))
        gt["y_max"].append(int(random.uniform(gt["y_min"][-1], 200)))
gt_df = pd.DataFrame(gt)
gt_df.to_csv("gt.csv", index=False)
print(gt_df.head())
predictions_df = gt_df.copy()

##Add noise into gt to create dummy predictions
predictions_df["class_id"] = predictions_df["class_id"].apply(lambda x: x if random.random() < predicted_acc else random.randint(0, 106))
predictions_df["confidence_score"] = predictions_df["confidence_score"].apply(lambda x: random.uniform(0.7, 1))
predictions_df["x_min"] = predictions_df["x_min"].apply(lambda x: int(max(0,min(random.uniform(x-20, x+20), 200))))
predictions_df["x_max"] = predictions_df["x_max"].apply(lambda x: int(max(0,min(random.uniform(x-20, x+20), 200))))
predictions_df["y_min"] = predictions_df["y_min"].apply(lambda x: int(max(0,min(random.uniform(x-20, x+20), 200))))
predictions_df["y_max"] = predictions_df["y_max"].apply(lambda x: int(max(0,min(random.uniform(x-20, x+20), 200))))
predictions_df.to_csv("predictions.csv", index=False)

unique_id = gt_df["image_name"].unique().tolist()

##Add extra prediction box per image 
for image_id in unique_id:
    if random.random()<extra_box_rate:
        xmin = int(random.uniform(0, 200))
        ymin = int(random.uniform(0, 200))
        gt_df = gt_df.append({"image_name": image_id, "class_id": random.randint(0, 106), "confidence_score": random.uniform(0.6, 1), 
                                "x_min": xmin, "x_max": int(random.uniform(xmin, 200)), 
                                "y_min": ymin, "y_max": int(random.uniform(ymin, 200))}, ignore_index=True)

print(predictions_df.head())
