import os
import cv2
import glob
import tqdm
import numpy as np
from Ai4MarsUtils import err, gray2color_s
#from sklearn.metrics import jaccard_score


IMAGES_PATH = "dataset/ai4mars-dataset-merged-0.3/msl/images/edr/"
LABELS_PATH = "dataset/ai4mars-dataset-merged-0.3/msl/labels/train/"
MASK_PATH = "dataset/ai4mars-dataset-unmerged/msl/train/"
OUTPUT_PATH = "dataset/NEW_MERGED/"

LABEL = [0,1,2,3,255]

AGREEMENT = 0.65
NUM_PIXELS = 1048576                                                    # 1024x1024
THRESHOLD_IMG = 2
def mergeRule(masks: list[np.array]) -> np.array:
    if len(masks) <= 2:
        err("\nError: No masks to merge")
        return np.array([])
    
    output_mask = np.full(masks[0].shape, 255, dtype=np.uint8)          # define output mask

    for i in range(masks[0].shape[0]):
        for j in range(masks[0].shape[1]):
            
            occurrences_map = { l:0 for l in LABEL }                    # count occurrences of each label
            for mask in masks:
                occurrences_map[mask[i][j]] += 1
            
            occurrences_map.pop(255)                                    # remove 255 key values
            
            max_key = max(occurrences_map, key=occurrences_map.get)     # get key with max value
            num_lbl = sum(occurrences_map.values())                     # get total number of labels
            
            if occurrences_map[max_key] < THRESHOLD_IMG: continue                   # first rule: at least 3 labelers agree
            if occurrences_map[max_key]/num_lbl < AGREEMENT: continue   # second rule: at least 65% of labelers agree
            
            output_mask[i,j] = max_key
    
    return output_mask
    
def main():
    
    
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    images = os.listdir(IMAGES_PATH)                                # get all images names
    
    count = 0
    for file in tqdm.tqdm(images):
        count += 1
        if count < 280: continue
        masks_paths = glob.glob(MASK_PATH + file.replace('.JPG','') + '*.png')
        if len(masks_paths) < 1: continue
        
        masks =[]
        for i in range(len(masks_paths)):
            mask = cv2.imread(masks_paths[i], cv2.IMREAD_GRAYSCALE)
            masks.append(mask)
        
        # save new mask
        new_mask = mergeRule(masks)
        #check if new mask is empty
        if new_mask.size == 0: continue
        cv2.imwrite(OUTPUT_PATH + file.replace('.JPG','_merged.png'), new_mask)

        # check differences
        ground_truth = cv2.imread(LABELS_PATH+file.replace('.JPG','') + '.png', cv2.IMREAD_GRAYSCALE)
        diff = cv2.absdiff(new_mask, ground_truth)
        if np.any(diff):
            print("\nDifferences found in ", file)
        
        #cv2.imshow('ground_truth', gray2color_s(ground_truth))
        #cv2.imshow('new_mask', gray2color_s(new_mask))
        #cv2.waitKey(0)
        #cv2.imshow('diff', diff)
        #cv2.waitKey(0)
        
        # calculate IoU multicalss
        #iou = jaccard_score(ground_truth.flatten(), new_mask.flatten(), average='macro')
        #print("\nIoU is ", iou)

    return

if __name__ == "__main__":
    main()
