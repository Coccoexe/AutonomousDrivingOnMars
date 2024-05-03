import os
import shutil

# CONSTANTS

DATA_FOLDER = os.getcwd() + "/dataset/"
DATASET_PATH = DATA_FOLDER + "ai4mars-dataset-merged-0.3/"
IMAGES_PATH = DATASET_PATH + "msl/images/edr/"
LABELS_PATH = DATASET_PATH + "msl/labels/"
OUTPUT_PATH = DATA_FOLDER + "ai4mars_ORIGINAL/"

def rearrange_dataset() -> None:
    print("Rearranging dataset...")

    # get list of files
    train = [file.replace('.png','') for file in os.listdir(LABELS_PATH + "train") if file.endswith(".png")]
    test  = [file.replace('_merged.png','') for file in os.listdir(LABELS_PATH + "test/masked-gold-min1-100agree") if file.endswith(".png")]

    images = OUTPUT_PATH + "/images/"
    labels = OUTPUT_PATH + "/labels/"
    if not os.path.exists(images + "train/"):
        os.makedirs(images + "train/")
    if not os.path.exists(images + "test/"):
        os.makedirs(images + "test/")
    if not os.path.exists(labels + "train/"):
        os.makedirs(labels + "train/")
    if not os.path.exists(labels + "test/"):
        os.makedirs(labels + "test/")

    for file in train:
        if not os.path.exists(IMAGES_PATH + file + ".jpg"): continue
        shutil.copy(IMAGES_PATH + file + ".jpg", images + "train/" + file + ".jpg")
        shutil.copy(LABELS_PATH + "train/" + file + ".png", labels + "train/" + file + ".png")
    for file in test:
        if not os.path.exists(IMAGES_PATH + file + ".jpg"): continue
        shutil.copy(IMAGES_PATH + file + ".jpg", images + "test/" + file + ".jpg")
        shutil.copy(LABELS_PATH + "test/masked-gold-min1-100agree/" + file + "_merged.png", labels + "test/" + file + ".png")

    print("Dataset rearranged.")
    return

def main():

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    # Rearrange dataset
    rearrange_dataset()

    return

if __name__ == "__main__":
    main()