import pandas as pd
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from PIL import Image

# Load the annotations for training and validation from separate CSV files
IMAGE_FOLDER = "/data/AffectNet/train_set/images/"
IMAGE_FOLDER_TEST = "/data/AffectNet/val_set/images/"
valid_annotations_path = (
    "../../affectnet_annotations/val_set_annotation_without_lnd.csv"
)
valid_annotations_df = pd.read_csv(valid_annotations_path)
valid_annotations_df = valid_annotations_df[valid_annotations_df["exp"] != 7]
# Set parameters
BATCHSIZE = 128
MODEL = models.maxvit_t(weights="DEFAULT")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# **** Create dataset and data loaders ****
class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, balance=False):
        self.dataframe = dataframe
        self.transform = transform
        self.root_dir = root_dir
        self.balance = balance

        if self.balance:
            self.dataframe = self.balance_dataset()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = os.path.join(
            self.root_dir, f"{self.dataframe['number'].iloc[idx]}.jpg"
        )
        if os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            image = Image.new(
                "RGB", (224, 224), color="white"
            )  # Handle missing image file

        label = torch.tensor(self.dataframe["exp"].iloc[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

    def balance_dataset(self):
        balanced_df = self.dataframe.groupby("exp", group_keys=False).apply(
            lambda x: x.sample(self.dataframe["exp"].value_counts().min())
        )
        return balanced_df


transform_valid = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
valid_dataset = CustomDataset(
    dataframe=valid_annotations_df,
    root_dir=IMAGE_FOLDER_TEST,
    transform=transform_valid,
    balance=False,
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=48
)
# ***** Define the model *****

# Initialize the model
block_channels = MODEL.classifier[3].in_features
MODEL.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.LayerNorm(block_channels),
    nn.Linear(block_channels, block_channels),
    nn.Tanh(),
    nn.Linear(block_channels, 7, bias=False),
)
MODEL.to(DEVICE)  # Put the model to the GPU

# Set the model to evaluation mode
MODEL.load_state_dict(torch.load("model.pt"))
MODEL.to(DEVICE)
MODEL.eval()

all_labels_cls = []
all_predicted_cls = []

# Start inference on test set
with torch.no_grad():
    for images, labels_cls in iter(valid_loader):
        images = images.to(DEVICE)
        labels_cls = labels_cls.to(DEVICE)

        outputs = MODEL(images)

        _, predicted_cls = torch.max(outputs, 1)

        all_labels_cls.extend(labels_cls.cpu().numpy())
        all_predicted_cls.extend(predicted_cls.cpu().numpy())


df = pd.DataFrame({"cat_pred": all_predicted_cls, "cat_true": all_labels_cls})
df.to_csv("inference.csv", index=False)
