import albumentations, cv2, os

def augment(image, mask):
    aug = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Rotate(p=0.5),
        albumentations.RandomCrop(p=0.75, height=512, width=512)])
    augmented = aug(image=image, mask=mask)
    return augmented["image"], augmented["mask"]

def main():

    images_path = "dataset/ai4mars-dataset-merged-0.4/images/train/"
    labels_path = "dataset/ai4mars-dataset-merged-0.4/labels/train/"

    if not os.path.exists("src/dataaugmentation/output/"):
        os.makedirs("src/dataaugmentation/output/")
    if not os.path.exists("src/dataaugmentation/output/images/"):
        os.makedirs("src/dataaugmentation/output/images/")
    if not os.path.exists("src/dataaugmentation/output/labels/"):
        os.makedirs("src/dataaugmentation/output/labels/")

    for file in open("src/dataaugmentation/rocks.txt", "r"):
        name = file.strip()
        image = cv2.imread(images_path + name + ".jpg", cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(labels_path + name + ".png", cv2.IMREAD_GRAYSCALE)
        augmented_image, augmented_mask = augment(image, mask)
        if augmented_image.shape != image.shape:
            if 3 not in augmented_mask:
                continue
            print(name)
            augmented_image = cv2.resize(augmented_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            augmented_mask = cv2.resize(augmented_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        cv2.imwrite("src/dataaugmentation/output/images/" + name + "_aug.jpg", augmented_image)
        cv2.imwrite("src/dataaugmentation/output/labels/" + name + "_aug.png", augmented_mask)




if __name__ == "__main__":
    main()
