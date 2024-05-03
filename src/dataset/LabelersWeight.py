import os
import cv2
import glob
import tqdm
import numpy as np

IMAGES_PATH = "dataset/ai4mars-dataset-merged-0.3/msl/images/edr/"
LABELS_PATH = "dataset/ai4mars-dataset-merged-0.3/msl/labels/train/"
MASK_PATH = "dataset/ai4mars-dataset-unmerged/msl/train/"

LABEL = [0,1,2,3,255]

def dice_coeff_siglelbl(label_true, label_pred, lbl):
    if not (np.any(label_true == lbl) and np.any(label_pred == lbl)):
        return 0
    intersection = np.sum((label_true == lbl) & (label_pred == lbl))
    dice_coefficient = (2.0 * intersection) / (np.sum(label_true == lbl) + np.sum(label_pred == lbl))
    return dice_coefficient

def dice_coeff(label_true, label_pred):
    dice_losses = []
    for l in LABEL:

        dice_loss = dice_coeff_siglelbl(label_true, label_pred, l)
        dice_losses.append(dice_loss)
    return dice_losses

    
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
            dice_losses = dice_coeff(ground_truth, mask)
            if labeler not in labelers:
                labelers[labeler] = [diff, 1, dice_losses]
            else:
                labelers[labeler][0] += diff
                labelers[labeler][1] += 1
                labelers[labeler].append(dice_losses)
    
    for key in labelers:
        total_diff = labelers[key][0]/(labelers[key][1]/(1024*1024))
        dice_loss_total = []
        for i in range(len(labelers[key][2])):
            dice_loss_total.append(labelers[key][2][i]/labelers[key][1])

        labelers[key] = total_diff, dice_loss_total
    
    with open('src/dataset/files/labelers_weight.txt', 'w') as f:
        for key in labelers:
            f.write(key+ ';' + str(labelers[key][0]))
            for i in range(len(labelers[key][1])):
                f.write(';' + str(labelers[key][1][i]))
            f.write('\n')
     
    return

if __name__ == "__main__":
    main()
