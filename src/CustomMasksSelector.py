import os
import cv2
import glob
import tqdm
import numpy as np

IMAGES_PATH = "dataset/ai4mars-dataset-merged-0.3/mer/images/eff/"
MASK_PATH = "dataset/ai4mars-dataset-merged-0.3/mer/labels/train/merged-unmasked/"

COLOR_NAMES = {0: 'soil', 1: 'bedrock', 2: 'sand', 3: 'big rocks', 255: 'no label'}
COLOR_MAP = {0: [0, 0, 0], 1: [128, 128, 128], 2: [0, 165, 255], 3: [0, 0, 255], 255: [255, 255, 255]}

OUTPUT_PATH = "dataset/MASKS/"

KEEP_PROGRESS = True
REVERSE_ORDER = True

def main():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # LEGEND -----------------------------------------------------------------------------------------------------
    legend = np.zeros((len(COLOR_MAP) * 22, 300, 3), dtype=np.uint8)
    i = 0
    for k, v in COLOR_MAP.items():
        cv2.putText(legend, COLOR_NAMES[k], (10, 20+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(legend, (150, i*20), (300, (i+1)*20), tuple(v), -1)
        i += 1
    cv2.imshow('legend', legend)


    # GET IMAGES ------------------------------------------------------------------------------------------------
    images = os.listdir(IMAGES_PATH)
    print('Total images:', len(images))
    if KEEP_PROGRESS:
        images = [i for i in images if not os.path.exists(OUTPUT_PATH + i)]
        print('Remaining images:', len(images))
    if REVERSE_ORDER:
        images = list(reversed(images))
    
    # LOOP IMAGES -----------------------------------------------------------------------------------------------
    for file in tqdm.tqdm(images):
        masks = glob.glob(MASK_PATH + file.replace('.JPG','') + '*.png')
        if len(masks) < 1: continue
        if len(masks) == 1:
            cv2.imwrite(OUTPUT_PATH + file, cv2.imread(masks[0], cv2.IMREAD_GRAYSCALE))
            continue

        img = cv2.imread(IMAGES_PATH + file, cv2.IMREAD_COLOR)                            

        # COLLAGE OF ALL MASKS ----------------------------------------------------------------------------------------------
        all_masks = np.zeros((img.shape[0], img.shape[1]*len(masks),3), dtype=np.uint8)
        for i in range(len(masks)):
            mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)

            rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            for k, v in COLOR_MAP.items():
                rgb_mask[mask == k] = v

            all_masks[:, i*img.shape[1]:(i+1)*img.shape[1]] = rgb_mask

            cv2.putText(all_masks, str(i+1), (i*img.shape[1]+5, img.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 0), 6)

        if len(masks) > 1:
            all_masks = cv2.resize(all_masks, (img.shape[1]*2, 2*img.shape[0]//len(masks)))     # resize to fit in the screen
        
        cv2.imshow('img', img)
        cv2.imshow('mask', all_masks)

        # SAVE SELECTED MASK -----------------------------------------------------------------------------------------------
        input = 0
        while input < 49 or input > 48+len(masks):
            input = cv2.waitKey(0)
            if input < 49 or input > 48+len(masks):
                print('Invalid input. Please press a number between 1 and', len(masks))
        cv2.imwrite(OUTPUT_PATH + file, cv2.imread(masks[input-49], cv2.IMREAD_GRAYSCALE))

        cv2.destroyWindow('img')
        cv2.destroyWindow('mask')

    return

if __name__ == "__main__":
    main()
