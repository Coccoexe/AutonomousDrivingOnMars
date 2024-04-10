import os
import cv2
import glob
import tqdm
import numpy as np

IMAGES_PATH = "dataset/ai4mars-dataset-merged-0.3/msl/images/edr/"
LABELS_PATH = "dataset/ai4mars-dataset-merged-0.3/msl/labels/train/"
MASK_PATH = "dataset/ai4mars-dataset-unmerged/msl/train/"
OUTPUT_PATH = "dataset/NEW_ROCKS/"

ROVER_MASKS = "dataset/ai4mars-dataset-merged-0.3/msl/images/mxy"
HORIZON_MASKS ="dataset/ai4mars-dataset-merged-0.3/msl/images/rng-30m"

PERCENT_SINGLE = 0.7
NUM_PIXELS = 1048576 # 1024x1024
EPSILON = 0.2

# DEBUG -----------------------------------------------------------------------------------------------------

COLOR_NAMES = {0: 'soil', 1: 'bedrock', 2: 'sand', 3: 'big rocks', 255: 'no label'}
COLOR_MAP = {0: [0, 0, 0], 1: [128, 128, 128], 2: [0, 165, 255], 3: [0, 0, 255], 255: [255, 255, 255]}

legend = np.zeros((len(COLOR_MAP) * 22, 300, 3), dtype=np.uint8)
i = 0
for k, v in COLOR_MAP.items():
    cv2.putText(legend, COLOR_NAMES[k], (10, 20+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(legend, (150, i*20), (300, (i+1)*20), tuple(v), -1)
    i += 1

def debug(imgs):
    for i in range(len(imgs)):
        img_color = np.zeros((imgs[i].shape[0], imgs[i].shape[1], 3), dtype=np.uint8)
        for k, v in COLOR_MAP.items():
            img_color[imgs[i] == k] = v
        cv2.imshow(str(i), img_color)
    cv2.imshow('legend', legend)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

# MAIN -----------------------------------------------------------------------------------------------------
    
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

    # GET IMAGES ------------------------------------------------------------------------------------------------
    images = os.listdir(IMAGES_PATH)
    
    # LOOP IMAGES -----------------------------------------------------------------------------------------------
    for file in tqdm.tqdm(images):

        filename = file.replace('.JPG','')
        masks = glob.glob(MASK_PATH + filename + '*.png')
        if len(masks) < 1: continue
        img = cv2.imread(IMAGES_PATH + file, cv2.IMREAD_COLOR)
        ground_truth = cv2.imread(LABELS_PATH + filename + '.png', cv2.IMREAD_GRAYSCALE)

        rocks_masks = []
        for i in range(len(masks)):
            mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
            if 3 in mask: rocks_masks.append(mask)
        
        if len(rocks_masks) < 1: continue

        if len(rocks_masks) == 1:
            if not single_mask_extraction(rocks_masks[0], ground_truth): continue
        else:
            multiple_mask_extraction(rocks_masks, ground_truth)
        
        # merge with rover and horizon masks
        rover_mask = cv2.imread(ROVER_MASKS + filename + '.png', cv2.IMREAD_GRAYSCALE)
        horizon_mask = cv2.imread(HORIZON_MASKS + filename + '.png', cv2.IMREAD_GRAYSCALE)
        merge_mask(ground_truth, rover_mask, horizon_mask)

        # save new ground truth
        cv2.imwrite(OUTPUT_PATH + filename + '.png', ground_truth)

    return

if __name__ == "__main__":
    main()
