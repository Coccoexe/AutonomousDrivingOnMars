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

    return

if __name__ == "__main__":
    main()