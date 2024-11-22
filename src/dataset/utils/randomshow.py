import cv2, os, numpy as np, tqdm

images = os.listdir('NEW_MASKS/')
for img in tqdm.tqdm(images):
    if np.random.rand() < 0.001:
        img = cv2.imread('NEW_MASKS/' + img)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()