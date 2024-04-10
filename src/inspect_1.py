import os
import cv2
import glob
import tqdm
import numpy as np
from Ai4MarsUtils import err, gray2color_s
#from sklearn.metrics import jaccard_score

wd = os.getcwd()
wd = os.path.dirname(wd)
wd = os.path.dirname(wd)


IMAGES_PATH = "dataset/ai4mars-dataset-merged-0.3/msl/images/edr/"
LABELS_PATH = "dataset/ai4mars-dataset-merged-0.3/msl/labels/train/"
MASK_PATH = "dataset/ai4mars-dataset-unmerged/msl/train/"
OUTPUT_PATH = "dataset/NEW_MERGED/"

    
def main():
  
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    images = os.listdir(IMAGES_PATH)                                # get all images names
    
    count = 0
    for file in tqdm.tqdm(images):
        count += 1
        if count < 281: continue
        if not os.path.exists(OUTPUT_PATH+file.replace('.JPG','') + '_merged.png'):
            continue
        
        new_mask = cv2.imread(OUTPUT_PATH+file.replace('.JPG','') + '_merged.png', cv2.IMREAD_GRAYSCALE)
        
        ground_truth = cv2.imread(LABELS_PATH+file.replace('.JPG','') + '.png', cv2.IMREAD_GRAYSCALE)

        
        cv2.imshow('ground_truth', gray2color_s(ground_truth))
        cv2.imshow('new_mask', gray2color_s(new_mask))
        cv2.waitKey(0)
        #cv2.imshow('diff', diff)
        #cv2.waitKey(0)
        
        # calculate IoU multicalss
        #iou = jaccard_score(ground_truth.flatten(), new_mask.flatten(), average='macro')
        #print("\nIoU is ", iou)

    return

if __name__ == "__main__":
    main()
