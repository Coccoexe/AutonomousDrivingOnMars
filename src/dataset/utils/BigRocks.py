import os
import cv2
import tqdm
import numpy as np

IMAGES_PATH = "C:/Users/alessio/Desktop/Tesi/Code/dataset/ai4mars-dataset-merged-0.3/msl/images/edr/"
MASK_PATH = "C:/Users/alessio/Desktop/Tesi/Code/dataset/ai4mars-dataset-merged-0.3/msl/labels/train/"
OUT_PATH = "./BIG_ROCKS/"

COLOR_NAMES = {0: 'soil', 1: 'bedrock', 2: 'sand', 3: 'big rocks', 255: 'no label'}
COLOR_MAP = {0: [0, 0, 0], 1: [128, 128, 128], 2: [0, 165, 255], 3: [0, 0, 255], 255: [255, 255, 255]}

def main():
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    #legend = np.zeros((len(COLOR_MAP) * 22, 1024, 3), dtype=np.uint8)
    legend = np.full((len(COLOR_MAP) * 22 + 20, 1024, 3), 20, dtype=np.uint8)
    cv2.putText(legend, 'Mask Color legend:', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    i = 0
    for k, v in COLOR_MAP.items():
        cv2.putText(legend, COLOR_NAMES[k], (30, 40+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(legend, (180, 40+i*20), (300, (i+1)*20), tuple(v), -1)
        cv2.rectangle(legend, (180, 40+i*20), (300, (i+1)*20), (255, 255, 255), 1)
        i += 1

    for file in tqdm.tqdm(os.listdir(MASK_PATH)):
        if file.endswith(".png"):
            mask = cv2.imread(MASK_PATH + file, cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(IMAGES_PATH + file.replace('png','jpg'), cv2.IMREAD_COLOR)

            # overlap the mask with the image
            if 3 in mask:
                size = [512, 512]
                mask = cv2.resize(mask, (size[1], size[0]))
                img = cv2.resize(img, (size[1], size[0]))
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                new_mask = np.zeros_like(mask)
                new_mask[np.where((mask == [0, 0, 0]).all(axis=2))] = COLOR_MAP[0]
                new_mask[np.where((mask == [1, 1, 1]).all(axis=2))] = COLOR_MAP[1]
                new_mask[np.where((mask == [2, 2, 2]).all(axis=2))] = COLOR_MAP[2]
                new_mask[np.where((mask == [3, 3, 3]).all(axis=2))] = COLOR_MAP[3]
                new_mask[np.where((mask == [255, 255, 255]).all(axis=2))] = COLOR_MAP[255]
                mask = cv2.addWeighted(img, 0.6, new_mask, 0.5, 0)

                final_image = np.concatenate((img, mask), axis=1)
                final_image = np.concatenate((final_image, legend), axis=0)

                cv2.imwrite(OUT_PATH + file, final_image)
    return

if __name__ == "__main__":
    main()