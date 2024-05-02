import os
import cv2
import glob
import tqdm
import numpy as np
from Ai4MarsUtils import err, gray2color

wd = os.getcwd()
wd = os.path.dirname(wd)
wd = os.path.dirname(wd)


IMAGES_PATH = "dataset/ai4mars_preprocessed_ai4mars_ORIGINAL/images/test/"
LABELS_PATH = "dataset/ai4mars_preprocessed_ai4mars_ORIGINAL/labels/test/"
OUTPUT_PATH = "predictions/deeplabv3plus_resnet18_240501-1607/semanticsegOutput/"
    
def main():
  
    images = os.listdir(OUTPUT_PATH)                                # get all images names
    
    for file in tqdm.tqdm(images):
        name = file.split('_')[-1]
        image = cv2.imread(IMAGES_PATH+name)
        gt = cv2.imread(LABELS_PATH+name, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(OUTPUT_PATH+file, cv2.IMREAD_GRAYSCALE)

        pred[pred == 1] = 0
        pred[pred == 2] = 1
        pred[pred == 3] = 2
        pred[pred == 4] = 3
        pred[pred == 5] = 255
        
        cv2.imshow('image', image)
        cv2.imshow('ground_truth', gray2color(gt))
        cv2.imshow('pred', gray2color(pred))
        cv2.waitKey(0)

    return

if __name__ == "__main__":
    main()
