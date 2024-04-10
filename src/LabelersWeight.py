import os
import cv2
import glob
import tqdm
import numpy as np

IMAGES_PATH = "dataset/ai4mars-dataset-merged-0.3/msl/images/edr/"
LABELS_PATH = "dataset/ai4mars-dataset-merged-0.3/msl/labels/train/"
MASK_PATH = "dataset/ai4mars-dataset-unmerged/msl/train/"


    
def main():

    images = os.listdir(IMAGES_PATH)                                # get all images names
    
    labelers = {}

    for file in tqdm.tqdm(images):
        filename = file.replace('.JPG','')
        if not os.path.exists(LABELS_PATH + filename + '.png'): continue
        ground_truth = cv2.imread(LABELS_PATH+ filename + '.png', cv2.IMREAD_GRAYSCALE)
        masks = glob.glob(MASK_PATH + filename + '*.png')

        for i in range(len(masks)):
            labeler = masks[i].split('_')[-1][:-4]
            mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
            diff = np.sum(ground_truth != mask)
            if labeler not in labelers:
                labelers[labeler] = [diff, 1]
            else:
                labelers[labeler][0] += diff
                labelers[labeler][1] += 1
    
    for key in labelers:
        labelers[key] = labelers[key][0]/labelers[key][1]
    
    with open('labelers_weight.txt', 'w') as f:
        for key in labelers:
            f.write(key + ';' + str(labelers[key]) + '\n')

    return

if __name__ == "__main__":
    main()
