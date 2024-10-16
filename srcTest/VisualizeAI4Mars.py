import os
import cv2
import glob
import tqdm
import numpy as np
'''
path_s5mars = os.path.join(os.path.join(os.getcwd()), 'dataset','S5Mars')
path_s5_images_train = os.path.join(path_s5mars, 'images', 'train')
path_s5_images_test = os.path.join(path_s5mars,'images','test')
path_s5_images = os.path.join(path_s5mars,'images')
path_s5_labels_train = os.path.join(path_s5mars, 'labels', 'train')
path_s5_labels_test = os.path.join(path_s5mars,'labels','test')'''

path_AI4Mars = os.path.join(os.path.join(os.getcwd()), 'dataset/ai4mars-dataset-merged-0.3')
path_AI4Mars_img = os.path.join(path_AI4Mars, 'msl/images/edr/')
path_AI4Mars_lbl_train = os.path.join(path_AI4Mars, 'msl/labels/train/')


print(path_AI4Mars_lbl_train)

IMAGES_PATH = path_AI4Mars_img
LABEL_PATH = path_AI4Mars_lbl_train
COLOR_NAMES = {0: 'soil', 1: 'bedrock', 2: 'sand', 3: 'big rocks', 255: 'no label'}
COLOR_MAP = {0: [128, 128, 128], 1: [128, 128, 128], 2: [0, 165, 255], 3: [0, 0, 255], 255: [255, 255, 255]}

CHECK_ROCKS = False
IMSHOW = True

def main():
  
    # LEGEND -----------------------------------------------------------------------------------------------------
    legend = np.zeros((len(COLOR_MAP) * 22, 300, 3), dtype=np.uint8)
    i = 0
    point_rect_begin = 225
    len_rect = 150
    for k, v in COLOR_MAP.items():
        cv2.putText(legend, COLOR_NAMES[k], (10, 20+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(legend, (point_rect_begin, i*20), (point_rect_begin+len_rect, (i+1)*20), tuple(v), -1)
        i += 1
    if IMSHOW: cv2.imshow('legend', legend)


    # GET IMAGES ------------------------------------------------------------------------------------------------
    images = os.listdir(IMAGES_PATH)
    labels = os.listdir(LABEL_PATH)
    # LOOP IMAGES -----------------------------------------------------------------------------------------------
    for file in tqdm.tqdm(labels):
        path_label = LABEL_PATH + "\\" + file
        path_img = IMAGES_PATH + "\\" + file.replace('.png','') + '.JPG'
        lbl = cv2.imread(path_label)
        img = cv2.imread(path_img)
        print("Unique values in lbl:", np.unique(lbl))
        if(len(np.unique(lbl))<4):
            continue
        # COLOR LABELS -----------------------------------------------------------------------------------------------
        lbl_color = np.zeros((lbl.shape[0], lbl.shape[1], 3), dtype=np.uint8)
        for k, v in COLOR_MAP.items():
            mask = lbl == k
            lbl_color[mask] = np.array(v, dtype=np.uint8)
    
            
        # OVERLAP ---------------------------------------------------------------------------------------------------
        overlap = cv2.addWeighted(img, 1, lbl_color, 0.25, 0.0)     

        collage = [img, overlap ,lbl_color]


        row_dim, col_dim = 256, 256
        rows, cols = 1,3

        collage_img = np.zeros((rows*(row_dim+20), cols*col_dim, 3), dtype=np.uint8)
        collage_img[:,:,:] = 128
        count = -2
        for i in range(rows):
            for j in range(cols):
                if i*cols+j >= len(collage): break
                b_next_row = i*(row_dim+20)
                e_next_row = i*(row_dim+20)+256
                collage_img[b_next_row:e_next_row, j*col_dim:(j+1)*col_dim] = cv2.resize(collage[i*cols+j], (256, 256))
                if count == -2:
                    text = 'Original'
                elif count == -1:
                    text = 'Overlap'
                else:
                    text = "Label"
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
