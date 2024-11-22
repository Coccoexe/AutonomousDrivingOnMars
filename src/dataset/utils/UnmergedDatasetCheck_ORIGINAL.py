import os
import cv2
import glob
import tqdm
import numpy as np

IMAGES_PATH = "dataset/msl/images/edr/"
MASK_PATH = "dataset/msl/train/"

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
        masks = glob.glob(MASK_PATH + file.replace('.JPG','') + '*.png')
        if len(masks) < 1: continue

        collage = []
        img = cv2.imread(IMAGES_PATH + file, cv2.IMREAD_COLOR)
        msk = cv2.imread("dataset/msl/labels/train/"+file.replace('.JPG','') + '.png', cv2.IMREAD_GRAYSCALE)
        collage.append(img)  

        msk_color = np.zeros((msk.shape[0], msk.shape[1], 3), dtype=np.uint8)
        for k, v in COLOR_MAP.items():
            msk_color[msk == k] = v
        collage.append(msk_color)         

        # COLLAGE OF ALL MASKS ----------------------------------------------------------------------------------------------
        rocks = False
        labelers = []
        for i in range(len(masks)):
            labelers.append(masks[i].split('_')[-1][:-4])
            mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
            if 3 in mask: rocks = True

            rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            for k, v in COLOR_MAP.items():
                rgb_mask[mask == k] = v

            collage.append(rgb_mask)

        if CHECK_ROCKS:
            if not rocks: continue

        row_dim, col_dim = 256, 256
        rows, cols = len(collage)//4, 4
        if len(collage) % 4 > 0: rows += 1

        collage_img = np.zeros((rows*(row_dim+20), cols*col_dim, 3), dtype=np.uint8)
        collage_img[:,:,:] = 128
        count = -2
        for i in range(rows):
            for j in range(cols):
                if i*cols+j >= len(collage): break
                b_next_row = i*(row_dim+20)
                e_next_row = i*(row_dim+20)+256
                collage_img[b_next_row:e_next_row, j*col_dim:(j+1)*col_dim] = cv2.resize(collage[i*cols+j], (256, 256))
                match(count):
                    case -2:
                        text = 'Original'
                    case -1:
                        text = 'Ground Truth'
                    case _:
                        text = labelers[count]
                collage_img = cv2.putText(collage_img, text, (j*col_dim, e_next_row +15 ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (88, 219, 255), 1)
                count += 1

        if IMSHOW:
            scale = 1.4
            collage_img = cv2.resize(collage_img, (int(collage_img.shape[1]*scale), int(collage_img.shape[0]*scale)))
            cv2.imshow('collage', collage_img)
            cv2.waitKey(0)
            cv2.destroyWindow('collage')
        else: 
            cv2.imwrite('dataset/ROCKS_COLLAGE/' + file, collage_img)
        
    if not IMSHOW: cv2.imwrite('dataset/ROCKS_COLLAGE/' + 'legend.png', legend)
    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    main()
