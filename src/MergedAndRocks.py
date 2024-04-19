import os
import cv2
import glob
import tqdm
import numpy as np
from Ai4MarsUtils import err, gray2color
from skimage.measure import label, regionprops

DEBUG = False
REVERSE = False
RESTART = False # CAREFUL: it takes a lot

IMAGES_PATH = "dataset/ai4mars-dataset-merged-0.3/msl/images/edr/"
LABELS_PATH = "dataset/ai4mars-dataset-merged-0.3/msl/labels/train/"
ROVER_MASKS = "dataset/ai4mars-dataset-merged-0.3/msl/images/mxy"
HORIZON_MASKS ="dataset/ai4mars-dataset-merged-0.3/msl/images/rng-30m"
MASK_PATH = "dataset/ai4mars-dataset-unmerged/msl/train/"
OUTPUT_PATH = "dataset/NEW_MERGED_OPT/"

LABEL = [0,1,2,3,255]

AGREEMENT = 0.55
NUM_PIXELS = 1048576                                                    # 1024x1024
THRESHOLD_IMG = 2
PERCENT_SINGLE = 0.7
EPSILON = 0.2

OUTLIERS = [line for line in open("outliers.txt", "r").readlines() if line != '']

def mergeRule(masks: list[np.array]) -> np.array:

    if len(masks) < THRESHOLD_IMG:
        #err("\nError: No masks to merge")
        if len(masks) != 0:
            ret = np.full(masks[0].shape, 255, dtype=np.uint8)
        else:
            ret = np.full((1024, 1024), 255, dtype=np.uint8)
        return ret
    
    output_mask = np.full(masks[0].shape, 255, dtype=np.uint8)          # define output mask

    for i in range(masks[0].shape[0]):
        for j in range(masks[0].shape[1]):
            
            occurrences_map = { l:0 for l in LABEL }                    # count occurrences of each label
            for mask in masks:
                occurrences_map[mask[i][j]] += 1
            
            occurrences_map.pop(255)                                    # remove 255 key values
            
            max_key = max(occurrences_map, key=occurrences_map.get)     # get key with max value
            num_lbl = sum(occurrences_map.values())                     # get total number of labels
            
            if occurrences_map[max_key] < THRESHOLD_IMG: continue       # first rule: at least 3 labelers agree
            if occurrences_map[max_key]/num_lbl < AGREEMENT: continue   # second rule: at least 65% of labelers agree
            
            output_mask[i,j] = max_key
    
    return output_mask

def OptimizedMergeRule(masks: list[np.array]) -> np.array:
    if len(masks) < THRESHOLD_IMG:
        #err("\nError: No masks to merge")
        if len(masks) != 0:
            ret = np.full(masks[0].shape, 255, dtype=np.uint8)
        else:
            ret = np.full((1024, 1024), 255, dtype=np.uint8)
        return ret
    
    output_mask = np.full(masks[0].shape, 255, dtype=np.uint8)          # define output mask

    occurrences_maps = np.zeros((len(masks),) + masks[0].shape + (len(LABEL),), dtype=np.uint8)
    for idx, mask in enumerate(masks):
        for i in range(len(LABEL)):
            occurrences_maps[idx,:,:,i] = mask == i

    occurrences_map_sum = np.sum(occurrences_maps, axis=0)
    total_labels = np.sum(occurrences_map_sum, axis=2)
    max_occurrences = np.max(occurrences_map_sum, axis=2)

    valid_pixels = np.logical_and(max_occurrences >= THRESHOLD_IMG, 
                               np.where(total_labels != 0, max_occurrences / total_labels >= AGREEMENT, False))
    output_mask[valid_pixels] = np.argmax(occurrences_map_sum, axis=2)[valid_pixels]
    
    return output_mask

def single_mask_extraction(mask, ground_truth) -> bool:
    equal_pixels = np.sum(mask == ground_truth)
    if (equal_pixels / NUM_PIXELS) < PERCENT_SINGLE:
        return False
    
    # extract rock
    ground_truth[mask == 3] = 3
    return True

def multiple_mask_extraction(imgs, ground_truth):
    
    n = len(imgs)
    rocks = np.zeros((imgs[0].shape[0], imgs[0].shape[1]), dtype=np.float32)
    for i in range(n):
        rocks[imgs[i] == 3] += (1/(n+EPSILON))
    
    ground_truth[rocks > np.random.rand(1)] = 3

    return True

def merge_mask(ground_truth, rover_mask, horizon_mask):
    ground_truth[rover_mask == 1] = 255
    ground_truth[horizon_mask == 1] = 255
    
    return True       
    
def main():
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    images = os.listdir(IMAGES_PATH)                                # get all images names
    done = os.listdir(OUTPUT_PATH)
    if not RESTART: images = [img for img in images if img.replace('.JPG','_merged.png') not in done]
    if REVERSE: images.reverse()
    
    for file in tqdm.tqdm(images):
        filename = file.replace('.JPG','')
        masks_paths = glob.glob(MASK_PATH + filename + '*.png')
        if len(masks_paths) < 1: continue
        ground_truth = cv2.imread(LABELS_PATH + filename + '.png', cv2.IMREAD_GRAYSCALE)
        
        masks =[]
        rocks_masks = []
        for i in range(len(masks_paths)):

            # check outliers
            if masks_paths[i].split('_')[-1].replace('.png','') in OUTLIERS: 
                continue

            mask = cv2.imread(masks_paths[i], cv2.IMREAD_GRAYSCALE)
            masks.append(mask)
            if 3 in mask: rocks_masks.append(i)

        #if len(rocks_masks) < 1: continue
        

        #new_mask = mergeRule(masks)    # merge masks
        new_mask = OptimizedMergeRule(masks)
    

        if len(rocks_masks) == 1:
            single_mask_extraction(masks[rocks_masks[0]], new_mask)
        elif len(rocks_masks) > 1:
            multiple_mask_extraction([masks[i] for i in rocks_masks], new_mask)


        #check if new mask is empty
        if new_mask.size == 0: continue

        if DEBUG:
            cv2.imshow("original mask", gray2color(ground_truth))
            cv2.imshow("new mask", gray2color(new_mask))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.imwrite(OUTPUT_PATH + file.replace('.JPG','_merged.png'), new_mask)
        
    return

if __name__ == "__main__":
    main()
