import os
import cv2
import glob
import tqdm
import numpy as np
from Ai4MarsUtils import err, gray2color

wd = os.getcwd()
wd = os.path.dirname(wd)
wd = os.path.dirname(wd)


IMAGES_PATH = "dataset/ai4mars-dataset-merged-0.4/images/test/"
LABELS_PATH = "dataset/ai4mars-dataset-merged-0.4/labels/test/"
    
def main():
  
    images = os.listdir(IMAGES_PATH)                                # get all images names
    
    for file in tqdm.tqdm(images):
        image = cv2.imread(IMAGES_PATH+file)
        gt = cv2.imread(LABELS_PATH+file, cv2.IMREAD_GRAYSCALE)
        
        cv2.imshow('image', image)
        cv2.imshow('ground_truth', gray2color(gt))
        cv2.waitKey(0)

    return

if __name__ == "__main__":
    main()
