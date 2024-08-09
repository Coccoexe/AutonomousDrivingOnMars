import cv2
import numpy as np
import os

 # 10 color classes
COLOR_MAP = {0: [0, 0, 0],1: [128, 128, 128],2: [0, 165, 255],3: [0, 0, 255],4: [255, 0, 0],5: [0, 255, 0],6: [255, 255, 0],7: [255, 0, 255],8: [0, 255, 255],9: [255, 255, 255]}


for image in os.listdir('src/unsupervised/output'):
    
    img = cv2.imread(f"src/unsupervised/output/{image}", cv2.IMREAD_GRAYSCALE)

    # make the image bigger
    img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_NEAREST)

    cv2.imshow('image', img)
    cv2.waitKey(0)

    # normalize in N classes
    N = 4
    img = img // (256 / N)

   
    colored = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for k, v in COLOR_MAP.items():
        colored[img == k] = v
    cv2.imshow('image', colored)
    cv2.waitKey(0)