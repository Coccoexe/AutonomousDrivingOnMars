import os, cv2, tqdm
path = "dataset/ai4mars-dataset-merged-0.3/"
images_path = path + "images/train/"
labels_path = path + "labels/train/"

images = os.listdir(images_path)
f = open("src/dataaugmentation/rocks.txt", "w")
for file in tqdm.tqdm(images):
    image = cv2.imread(images_path + file)
    gt = cv2.imread(labels_path + file.replace('jpg', 'png'), cv2.IMREAD_GRAYSCALE)
    if 3 in gt:
        f.write(file[:-4] + "\n")
f.close()