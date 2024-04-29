import os
import shutil
from sklearn import model_selection

# CONSTANTS
NAME_LABEL_FOLDER = "NEW_MERGED_OPT"

DATA_FOLDER = os.getcwd() + "/dataset/"

DATASET_PATH = DATA_FOLDER + "ai4mars-dataset-merged-0.3/"
IMAGES_PATH = DATASET_PATH + "msl/images/edr/"
LABELS_PATH = DATA_FOLDER + NAME_LABEL_FOLDER +"/"
OUTPUT_PATH = DATA_FOLDER + "ai4mars_data_prep_"+NAME_LABEL_FOLDER+"/"

def rearrange_dataset() -> None:
    print("Rearranging dataset...")

    # get list of files
    label = [file.replace('_merged.png','') for file in os.listdir(LABELS_PATH) if file.endswith(".png")]
    #split in train and test and sklean
    train, test = model_selection.train_test_split(label, test_size=0.2, random_state=42)
    print("Train size: ", len(train))
    print("Test size: ", len(test))


    images = OUTPUT_PATH + "/images/"
    labels = OUTPUT_PATH + "/labels/"
    if not os.path.exists(images + "train/"):
        os.makedirs(images + "train/")
    else:
        shutil.rmtree(images + "train/")
        os.makedirs(images + "train/")
    if not os.path.exists(images + "test/"):
        os.makedirs(images + "test/")
    else:
        shutil.rmtree(images + "test/")
        os.makedirs(images + "test/")
    if not os.path.exists(labels + "train/"):
        os.makedirs(labels + "train/")
    else:
        shutil.rmtree(labels + "train/")
        os.makedirs(labels + "train/")
    if not os.path.exists(labels + "test/"):
        os.makedirs(labels + "test/")
    else:
        shutil.rmtree(labels + "test/")
        os.makedirs(labels + "test/")

    for file in train:
        if not os.path.exists(IMAGES_PATH + file + ".JPG"): continue
        shutil.copy(IMAGES_PATH + file + ".JPG", images + "train/" + file + ".png")
        shutil.copy(LABELS_PATH + file + "_merged.png", labels + "train/" + file + ".png")
    for file in test:
        if not os.path.exists(IMAGES_PATH + file + ".JPG"): continue
        shutil.copy(IMAGES_PATH + file + ".JPG", images + "test/" + file + ".png")
        shutil.copy(LABELS_PATH + file + "_merged.png", labels + "test/" + file + ".png")

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
