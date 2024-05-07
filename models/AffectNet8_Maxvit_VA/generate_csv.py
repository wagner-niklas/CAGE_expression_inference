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

# Set parameters
BATCHSIZE = 128
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
        image = Image.open(image_path)

        classes = torch.tensor(self.dataframe.iloc[idx, 1], dtype=torch.int8)
        valence = torch.tensor(self.dataframe.iloc[idx, 2], dtype=torch.float16)
        arousal = torch.tensor(self.dataframe.iloc[idx, 3], dtype=torch.float16)

        if self.transform:
            image = self.transform(image)

        return image, classes, valence, arousal

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
MODEL = models.maxvit_t(weights="DEFAULT")
block_channels = MODEL.classifier[3].in_features
MODEL.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.LayerNorm(block_channels),
    nn.Linear(block_channels, block_channels),
    nn.Tanh(),
    nn.Dropout(0.3),
    nn.Linear(block_channels, 2, bias=False),
)
MODEL.to(DEVICE)

# **** Test the model performance for classification ****

# Set the model to evaluation mode
MODEL.load_state_dict(torch.load("model.pt"))
MODEL.to(DEVICE)
MODEL.eval()

all_val_true_values = []
all_val_predicted_values = []
all_aro_true_values = []
all_aro_predicted_values = []

# Start inference on test set
with torch.no_grad():
    for images, _, val_true, aro_true in valid_loader:
        images, val_true, aro_true = (
            images.to(DEVICE),
            val_true.to(DEVICE),
            aro_true.to(DEVICE),
        )

        outputs = MODEL(images)
        val_pred = outputs[:, 0]
        aro_pred = outputs[:, 1]

        # Append to the lists --> Regression
        true_val_values = val_true.cpu().numpy()
        true_aro_values = aro_true.cpu().numpy()
        pred_val_values = val_pred.cpu().numpy()
        pred_aro_values = aro_pred.cpu().numpy()
        all_val_true_values.extend(true_val_values)
        all_aro_true_values.extend(true_aro_values)
        all_val_predicted_values.extend(pred_val_values)
        all_aro_predicted_values.extend(pred_aro_values)
df = pd.DataFrame(
    {
        "val_pred": all_val_predicted_values,
        "val_true": all_val_true_values,
        "aro_pred": all_aro_predicted_values,
        "aro_true": all_aro_true_values,
    }
)
df.to_csv("inference.csv", index=False)
