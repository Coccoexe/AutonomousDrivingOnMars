import os
import zipfile
import urllib.request

# CONSTANTS

DATA_FOLDER = os.getcwd() + "/dataset/"
URL = "https://data.nasa.gov/download/cykx-2qix/application%2Fzip"
FILENAME = urllib.request.urlopen(URL).url.split("filename=")[-1]
DOWNLOAD_ATTEMPTS = 3

# FUNCTIONS

def download_dataset() -> None:
    print("Downloading dataset...")
    count = 0
    while count < DOWNLOAD_ATTEMPTS:
        count += 1
        try:
            print("Attempt", count)
            urllib.request.urlretrieve(URL, DATA_FOLDER + FILENAME)
            print("Dataset downloaded.")
        except Exception as e:
            print(e)
            if count == 3:
                print("Failed to download dataset. Please try again later.")
                os.remove(DATA_FOLDER + FILENAME)
    return

def unzip_dataset() -> None:
    print("Unzipping dataset...")
    with zipfile.ZipFile(DATA_FOLDER + FILENAME, 'r') as zip_ref:
        zip_ref.extractall(DATA_FOLDER)
    print("Dataset unzipped.")

# Note that this function is specific to the dataset used in this project
def rearrange_dataset() -> None:
    print("Rearranging dataset...")

    # get list of files
    train = [file.replace('.png','') for file in os.listdir(DATA_FOLDER + "ai4mars-dataset-merged-0.1/msl/labels/train") if file.endswith(".png")]
    test  = [file.replace('_merged.png','') for file in os.listdir(DATA_FOLDER + "ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min1-100agree") if file.endswith(".png")]

    images_path = DATA_FOLDER + "/ai4mars-dataset-merged-0.1/msl/images/edr/"
    if not os.path.exists(images_path + "train/"):
        os.makedirs(images_path + "train/")
    if not os.path.exists(images_path + "test/"):
        os.makedirs(images_path + "test/")

    for file in train:
        if not os.path.exists(images_path + file + ".jpg"): continue
        os.rename(images_path + file + ".jpg", images_path + "train/" + file + ".jpg")
    for file in test:
        if not os.path.exists(images_path + file + ".jpg"): continue
        os.rename(images_path + file + ".jpg", images_path + "test/" + file + ".jpg")

    print("Dataset rearranged.")
    return

def main():

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    # Download dataset
    if not os.path.exists(DATA_FOLDER + FILENAME):
        download_dataset()
    else:
        print("Dataset already downloaded, avoiding download.")

    # Unzip dataset
    if not os.path.exists(DATA_FOLDER + FILENAME.rstrip(".zip")):
        unzip_dataset()
    else:
        print("Dataset already unzipped, avoiding unzip.")

    # Rearrange dataset
    rearrange_dataset()

    return

if __name__ == "__main__":
    main()