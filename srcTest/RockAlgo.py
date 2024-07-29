import os
import cv2
import glob
import tqdm
import numpy as np

IMAGES_PATH = "dataset/ai4mars-dataset-merged-0.3/msl/images/edr/"
MASK_PATH = "dataset/ai4mars-dataset-unmerged/msl/train/"
GROUND_TRUTH_PATH = "dataset/ai4mars-dataset-merged-0.3/msl/labels/train/"

COLOR_NAMES = {0: 'soil', 1: 'bedrock', 2: 'sand', 3: 'big rocks', 255: 'no label'}
COLOR_MAP = {0: [0, 0, 0], 1: [128, 128, 128], 2: [0, 165, 255], 3: [0, 0, 255], 255: [255, 255, 255]}

CHECK_ROCKS = True
IMSHOW = False

def main():
    if not os.path.exists('dataset/ROCKS_COLLAGE/'):
            os.makedirs('dataset/ROCKS_COLLAGE/')

    # LEGEND -----------------------------------------------------------------------------------------------------
    legend = np.zeros((len(COLOR_MAP) * 22, 300, 3), dtype=np.uint8)
    i = 0
    for k, v in COLOR_MAP.items():
        cv2.putText(legend, COLOR_NAMES[k], (10, 20+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(legend, (150, i*20), (300, (i+1)*20), tuple(v), -1)
        i += 1
    if IMSHOW: cv2.imshow('legend', legend)


    # GET IMAGES ------------------------------------------------------------------------------------------------
    images = os.listdir(IMAGES_PATH)
    
    # LOOP IMAGES -----------------------------------------------------------------------------------------------
    for file in tqdm.tqdm(images):
        masks_path = glob.glob(MASK_PATH + file.replace('.JPG','') + '*.png')
        ground_truth_path = GROUND_TRUTH_PATH + file.replace('.JPG','') + '.png'
        
        if len(masks_path) < 1: continue
        #check if there are rocks labeled

        masks = []
        for i in range(len(masks_path)):
            mask = cv2.imread(masks_path[i], cv2.IMREAD_GRAYSCALE)
            if 3 in mask: masks.append(mask)
        #if there are no rocks labeled, continue
        if len(masks) <= 0: continue
        
        if len(masks) > 1:
            n= len(masks)
            #epsilon >= 0
            #create a image with the same size as the masks
            prob_image = np.zeros((masks[0].shape[0], masks[0].shape[1]), dtype=np.uint8)
            #cicle each pixel of the masks
            for i in range(masks[0].shape[0]):
                for j in range(masks[0].shape[1]):
                    #cicle each mask
                    for k in range(n):
                        #if the pixel is a rock, add 1 to the probability 
                        if masks[k][i][j] == 3:
                            prob_image[i][j] += 1
        


    return

if __name__ == "__main__":
    main()
