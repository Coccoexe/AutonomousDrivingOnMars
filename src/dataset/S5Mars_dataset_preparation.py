import os
import json
import shutil

# CONSTANTS

DATA_FOLDER = os.getcwd() + "/dataset/"
DATASET_PATH = DATA_FOLDER + "S5Mars_data/"
IMAGES_PATH = DATASET_PATH + "images/"
LABELS_PATH = DATASET_PATH + "labels/"
SPLIT_PATH = DATASET_PATH + "split/"
OUTPUT_PATH = DATA_FOLDER + "S5Mars/"

def rearrange_dataset() -> None:

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    if not os.path.exists(OUTPUT_PATH + "images/"):
        os.makedirs(OUTPUT_PATH + "images/")
    if not os.path.exists(OUTPUT_PATH + "labels/"):
        os.makedirs(OUTPUT_PATH + "labels/")
    if not os.path.exists(OUTPUT_PATH + "labels/train/"):
        os.makedirs(OUTPUT_PATH + "labels/train/")
    if not os.path.exists(OUTPUT_PATH + "labels/test/"):
        os.makedirs(OUTPUT_PATH + "labels/test/")
    if not os.path.exists(OUTPUT_PATH + "images/train/"):
        os.makedirs(OUTPUT_PATH + "images/train/")
    if not os.path.exists(OUTPUT_PATH + "images/test/"):
        os.makedirs(OUTPUT_PATH + "images/test/")

    for split in os.listdir(SPLIT_PATH):
        
        dir = split.split(".")[0] + "/"
        if dir == "val/":
            dir = "train/"

        with open(SPLIT_PATH + split, "r") as f:
            data = json.load(f)
        for img in data:
            shutil.copy(IMAGES_PATH + img + ".jpg", OUTPUT_PATH + "images/" + dir + img[5:] + ".jpg")
            shutil.copy(LABELS_PATH + img + ".png", OUTPUT_PATH + "labels/" + dir + img[5:] + ".png")        
    return

def main():
    rearrange_dataset()
    return

if __name__ == "__main__":
    main()