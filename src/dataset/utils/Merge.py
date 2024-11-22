import os
import cv2
import glob
import tqdm
import numpy as np

IMAGES_PATH = "C:/Users/alessio/Desktop/Tesi/Code/dataset/ai4mars-dataset-merged-0.3/msl/images/edr/"
MASK_PATH = "C:/Users/alessio/Desktop/Tesi/Code/dataset/ai4mars-dataset-merged-0.3/msl/labels/train/"
NEW_MASK_PATH = "C:/Users/alessio/Desktop/Tesi/Code/dataset/NEW_ROCKS/"
OUT_PATH = "NEW_MASKS/"

COLOR_NAMES = {0: 'soil', 1: 'bedrock', 2: 'sand', 3: 'big rocks', 255: 'no label'}
COLOR_MAP = {0: [0, 0, 0], 1: [128, 128, 128], 2: [0, 165, 255], 3: [0, 0, 255], 255: [255, 255, 255]}

IMSHOW = False

def main():
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    # LEGEND -----------------------------------------------------------------------------------------------------
    legend = np.zeros((len(COLOR_MAP) * 22, 300, 3), dtype=np.uint8)
    i = 0
    for k, v in COLOR_MAP.items():
        cv2.putText(legend, COLOR_NAMES[k], (10, 20+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(legend, (150, i*20), (300, (i+1)*20), tuple(v), -1)
        i += 1
    if IMSHOW: cv2.imshow('legend', legend)


    # GET IMAGES ------------------------------------------------------------------------------------------------
    images = os.listdir(NEW_MASK_PATH)
    
    # LOOP IMAGES -----------------------------------------------------------------------------------------------
    for file in tqdm.tqdm(images):
        collage = []
        img = cv2.imread(IMAGES_PATH + file.replace('.png','') + '.JPG', cv2.IMREAD_COLOR)
        msk = cv2.imread(MASK_PATH + file, cv2.IMREAD_GRAYSCALE)
        new_mask = cv2.imread(NEW_MASK_PATH + file, cv2.IMREAD_GRAYSCALE)
        collage.append(img)

        msk_color = np.zeros((msk.shape[0], msk.shape[1], 3), dtype=np.uint8)
        new_mask_color = np.zeros((new_mask.shape[0], new_mask.shape[1], 3), dtype=np.uint8)
        for k, v in COLOR_MAP.items():
            msk_color[msk == k] = v
            new_mask_color[new_mask == k] = v
        collage.append(msk_color)    
        collage.append(new_mask_color)

        collage_img = np.zeros((286,768 , 3), dtype=np.uint8)
        collage_img[:,:,:] = 128
        collage_img[0:256, 0:256] = cv2.resize(collage[0], (256, 256))
        collage_img[0:256, 256:512] = cv2.resize(collage[1], (256, 256))
        collage_img[0:256, 512:768] = cv2.resize(collage[2], (256, 256))
        collage_img = cv2.putText(collage_img, 'EDR image', (0, 276), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (88, 219, 255), 1)
        collage_img = cv2.putText(collage_img, 'Original mask', (256, 276), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (88, 219, 255), 1)
        collage_img = cv2.putText(collage_img, 'New mask', (512, 276), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (88, 219, 255), 1)

        if IMSHOW:
            scale = 1.4
            collage_img = cv2.resize(collage_img, (int(collage_img.shape[1]*scale), int(collage_img.shape[0]*scale)))
            cv2.imshow('collage', collage_img)
            cv2.waitKey(0)
            cv2.destroyWindow('collage')
        else: 
            cv2.imwrite(OUT_PATH + file, collage_img)
        
    if not IMSHOW: cv2.imwrite(OUT_PATH + 'legend.png', legend)
    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    main()
