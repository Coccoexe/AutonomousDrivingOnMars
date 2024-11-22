import os 
from tqdm import tqdm
import numpy as np
from PIL import Image as pilImage
from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
import numpy as np
from torch.utils.data import Dataset
import torch
import albumentations as A
from torch.utils.data import DataLoader
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput
import evaluate
from torch.optim import AdamW
from tqdm.auto import tqdm
from datasets import Dataset as d

BATCH_SIZE = 16
ADE_MEAN= [0.61993144, 0.51421535, 0.36058366]
ADE_STD= [0.2162113,  0.34357504, 0.55873912]

def load_image(dataset_path_img, dataset_path_lbl):
    image_files = [os.path.join(dataset_path_img, f) for f in os.listdir(dataset_path_img) if f.endswith(('.png', '.jpg', '.jpeg'))]
    label_files = [os.path.join(dataset_path_lbl, f) for f in os.listdir(dataset_path_lbl) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_to_label = {os.path.basename(img): lbl for img, lbl in zip(image_files, label_files)}

    print("example of the first path of an image file after shuffling: ", image_files[0])
    print("example of label file: ", label_files[0])

    # Define the classes
    classes = ["soil", "bedrock", "sand", "bigRock", "noLabel"]

    # Create a dataset from the image files
    features = Features({
        'image': Image(),
        'label': Image(),
        #'classes_on_image': Sequence(Value('int32')),
        #'id': Value('int32')
    })

    print(f"example correspondence {image_files[0]} example correspondence, {image_to_label[os.path.basename(image_files[0])]}")

    # Example data creation
    data_test = []
    for idx, img_path in enumerate(tqdm(image_files, desc="Processing images")):
        img_basename = os.path.basename(img_path)
        label_path = image_to_label[img_basename]
        
        #classes_on_image = extract_classes_on_image(label_path)
        data_test.append({
            'image': img_path,
            'label': label_path,
            #'classes_on_image': classes_on_image,
            #'id': idx
        })

    # Create the dataset
    dataset_S5Mars = DatasetDict({
        'test': d.from_list(data_test, features=features)
    })

    # Print the dataset
    print(dataset_S5Mars)
    return dataset_S5Mars

class SegmentationDataset(Dataset):
  def __init__(self, dataset, transform):
    self.dataset = dataset
    self.transform = transform

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    original_image = np.array(item["image"])
    original_segmentation_map = np.array(item["label"])

    transformed = self.transform(image=original_image, mask=original_segmentation_map)
    image, target = torch.tensor(transformed['image']), torch.LongTensor(transformed['mask'])

    # convert to C, H, W
    image = image.permute(2,0,1)

    return image, target, original_image, original_segmentation_map
def collate_fn(inputs):
    batch = dict()
    batch["pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["original_images"] = [i[2] for i in inputs]
    batch["original_segmentation_maps"] = [i[3] for i in inputs]

    return batch



class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2)

        return self.classifier(embeddings)


class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
  def __init__(self, config):
    super().__init__(config)

    self.dinov2 = Dinov2Model(config)
    self.classifier = LinearClassifier(config.hidden_size, 32, 32, config.num_labels)

  def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
    # use frozen features
    outputs = self.dinov2(pixel_values,
                            output_hidden_states=output_hidden_states,
                            output_attentions=output_attentions)
    # get the patch embeddings - so we exclude the CLS token
    patch_embeddings = outputs.last_hidden_state[:,1:,:]

    # convert to logits and upsample to the size of the pixel values
    logits = self.classifier(patch_embeddings)
    logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

    loss = None
    if labels is not None:
      # important: we're going to use 0 here as ignore index instead of the default -100
      # as we don't want the model to learn to predict background
      loss_fct = torch.nn.CrossEntropyLoss(ignore_index=255)
      loss = loss_fct(logits.squeeze(), labels.squeeze())

    return SemanticSegmenterOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    
def main():
    classes2names = {
        0:   "soil", 
        1:   "bedrock", 
        2:   "sand", 
        3:   "bigRock", 
        255: "noLabel"
    }

    test_transform = A.Compose([    
        A.Resize(width=448, height=448), #TODO: change
        A.Normalize(mean=ADE_MEAN, std=ADE_STD)
    ])
    
    dataset_path_img = ".\dataset/ai4mars-dataset-merged-0.4-preprocessed-512/images/test"
    dataset_path_lbl = ".\dataset/ai4mars-dataset-merged-0.4-preprocessed-512/labels/test"
    dataset_S5Mars = load_image(dataset_path_img, dataset_path_lbl)
    print(dataset_S5Mars)
    test_dataset = SegmentationDataset(dataset_S5Mars["test"], transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn) 
    
    model = Dinov2ForSemanticSegmentation.from_pretrained("microsoft/dinov2-base")
    model.load_state_dict(torch.load("./output/model_val.pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # predict and save images
    for i, batch in enumerate(test_dataloader):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        original_images = batch["original_images"]
        original_segmentation_maps = batch["original_segmentation_maps"]

        with torch.no_grad():
            outputs = model(pixel_values, labels=labels)
            predictions = torch.argmax(outputs.logits, dim=1)

        for original_image, original_segmentation_map, prediction in zip(original_images, original_segmentation_maps, predictions):
            original_image = pilImage.fromarray(original_image)
            original_segmentation_map = pilImage.fromarray(original_segmentation_map)
            prediction = pilImage.fromarray(prediction.cpu().numpy().astype(np.uint8))

            original_image.show()
            original_segmentation_map.show()
            prediction.show()

            break

        break

    


if __name__ == '__main__':
    main()