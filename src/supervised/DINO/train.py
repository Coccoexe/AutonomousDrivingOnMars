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




BATCH_SIZE = 8
ADE_MEAN= [0.61993144, 0.51421535, 0.36058366]
ADE_STD= [0.2162113,  0.34357504, 0.55873912]

learning_rate = 5e-5
epochs = 200


def load_image(dataset_path_img, dataset_path_lbl):
    image_files = [os.path.join(dataset_path_img, f) for f in os.listdir(dataset_path_img) if f.endswith(('.png', '.jpg', '.jpeg'))]
    label_files = [os.path.join(dataset_path_lbl, f) for f in os.listdir(dataset_path_lbl) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_to_label = {os.path.basename(img): lbl for img, lbl in zip(image_files, label_files)}

    #split image flies to train and val randomly
    np.random.seed(0)
    np.random.shuffle(image_files)
    image_files_train = image_files[:int(len(image_files)*0.8)]
    image_files_val = image_files[int(len(image_files)*0.8):]


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


    # Custom function to extract unique values from the label image
    def extract_classes_on_image(label_path):
        label_image = np.array(pilImage.open(label_path))
        return np.unique(label_image).tolist()

    # Example data creation
    data_train = []
    data_val = []
    for idx, img_path in enumerate(tqdm(image_files, desc="Processing images")):
        img_basename = os.path.basename(img_path)
        label_path = image_to_label[img_basename]
        
        
        #classes_on_image = extract_classes_on_image(label_path)
        if(img_path in image_files_train):
            data_train.append({
                'image': img_path,
                'label': label_path,
                #'classes_on_image': classes_on_image,
                #'id': idx
            })
        else:
            data_val.append({
                'image': img_path,
                'label': label_path,
                #'classes_on_image': classes_on_image,
                #'id': idx
            })

    # Create the dataset
    dataset_S5Mars = DatasetDict({
        'train': d.from_list(data_train, features=features),
        'validation': d.from_list(data_val, features=features)
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

    original_segmentation_map[original_segmentation_map == 255] = 4

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
      loss_fct = torch.nn.CrossEntropyLoss()
      loss = loss_fct(logits.squeeze(), labels.squeeze())

    return SemanticSegmenterOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    
    
def train(model, train_dataloader, val_dataloader, optimizer, metric, device, epochs,classes2names):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # put model in training mode
    model.train()
    traning_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = 0.0
        min_valloss = np.inf

        
        val_loss = 0.0
        miou_val = 0.0
        mean_accuracy_val = 0.0
        print("Epoch:", epoch)
        for idx, batch in enumerate(tqdm(train_dataloader)):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # forward pass
            outputs = model(pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()
            traning_losses.append(loss.item())
            train_loss += loss.item()
            
            
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_dataloader)):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                # forward pass
                outputs = model(pixel_values, labels=labels)
                loss = outputs.loss
                val_loss +=loss.item()
                
                predicted = outputs.logits.argmax(dim=1)
                metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
                metrics = metric.compute(num_labels=len(classes2names),
                                    ignore_index=None,
                                    reduce_labels=False,
                )
                miou_val += metrics["mean_iou"]
                mean_accuracy_val += metrics["mean_accuracy"]
        val_loss = val_loss / len(val_dataloader)
        miou_val = miou_val / len(val_dataloader)
        mean_accuracy_val = mean_accuracy_val / len(val_dataloader)
        train_loss = train_loss / len(train_dataloader)
        
        print(f"{'Train Loss:':<20} {train_loss:.4f}")
        print(f"{'Val Loss:':<20} {val_loss:.4f}")
        print(f"{'Val Mean IoU:':<20} {miou_val:.4f}")
        print(f"{'Val Mean Accuracy:':<20} {mean_accuracy_val:.4f}")
        
        if val_loss < min_valloss:
            print(f"Validation loss decreased from {min_valloss:.4f} to {val_loss:.4f}, saving model")
            min_valloss = val_loss
            torch.save(model.state_dict(), "./output/model_val.pth")
               
        val_losses.append(val_loss)

    return model, traning_losses, val_losses       

def main():
    classes2names = {
        0:   "soil", 
        1:   "bedrock", 
        2:   "sand", 
        3:   "bigRock", 
        4: "noLabel"
    }

    train_transform = A.Compose([    
        A.Resize(width=448, height=448), #TODO: change
        A.Normalize(mean=ADE_MEAN, std=ADE_STD)
    ])

    val_transform = A.Compose([
        A.Resize(width=448, height=448),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD)
    ])
    
    dataset_path_img = ".\dataset/ai4mars-dataset-merged-0.4-preprocessed-512/images/train"
    dataset_path_lbl = ".\dataset/ai4mars-dataset-merged-0.4-preprocessed-512/labels/train"
    dataset_S5Mars = load_image(dataset_path_img, dataset_path_lbl)
    print(dataset_S5Mars)
    train_dataset = SegmentationDataset(dataset_S5Mars["train"], transform=train_transform)
    val_dataset = SegmentationDataset(dataset_S5Mars["validation"], transform=val_transform)
    
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)    
    #debug
    batch = next(iter(train_dataloader))
    for k,v in batch.items():
        if isinstance(v,torch.Tensor):
            print(k,v.shape)
        else:
            print(k,len(v),v[0].shape)
            
    model = Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base", id2label=classes2names, num_labels=len(classes2names))
    for name, param in model.named_parameters():
        if name.startswith("dinov2"):
            param.requires_grad = False   

    metric = evaluate.load("mean_iou")
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # put model on GPU (set runtime to GPU in Google Colab)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    
    model, loss_train, loss_val = train(model, train_dataloader, val_dataloader, optimizer, metric, device, epochs,classes2names)

    #save model, loss_train, loss_val
    torch.save(model.state_dict(), "./output/model.pth")
    np.save("./output/loss_train.npy", loss_train)
    np.save("./output/loss_val.npy", loss_val)
    


if __name__ == '__main__':
    main()