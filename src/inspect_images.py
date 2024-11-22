import os
import cv2
import glob
import tqdm
import numpy as np
from Ai4MarsUtils import err, gray2color

IMAGES_PATH = "dataset/ai4mars-dataset-merged-0.3/images/test/"
LABELS_PATH = "dataset/ai4mars-dataset-merged-0.3/labels/test/"
OUTPUT_PATH = "dataset/ai4mars-dataset-merged-0.3/labels/test_inspected/"
DEBUG = False

def main():
  
    images = os.listdir(IMAGES_PATH)                                # get all images names
    
    for file in tqdm.tqdm(images):
        image = cv2.imread(IMAGES_PATH+file)
        gt = cv2.imread(LABELS_PATH+file.replace('jpg','png'), cv2.IMREAD_GRAYSCALE)
        
        if DEBUG:
            cv2.imshow('image', image)
            cv2.imshow('ground_truth', gray2color(gt))
            cv2.waitKey(0)
        
        else:
            cv2.imwrite(OUTPUT_PATH+file.replace('jpg','png'), gray2color(gt))
        

    return

if __name__ == "__main__":
    main()
