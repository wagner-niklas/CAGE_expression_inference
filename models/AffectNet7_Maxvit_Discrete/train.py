import pandas as pd
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision
from torch.optim import lr_scheduler
import re
from tqdm import tqdm

# Load the annotations for training and validation from separate CSV files
IMAGE_FOLDER = "/data/AffectNet/train_set/images/"
IMAGE_FOLDER_TEST = "/data/AffectNet/val_set/images/"
train_annotations_path = (
    "../../affectnet_annotations/train_set_annotation_without_lnd.csv"
)
valid_annotations_path = (
    "../../affectnet_annotations/val_set_annotation_without_lnd.csv"
)
train_annotations_df = pd.read_csv(train_annotations_path)
valid_annotations_df = pd.read_csv(valid_annotations_path)

train_annotations_df = train_annotations_df[train_annotations_df["exp"] != 7]
valid_annotations_df = valid_annotations_df[valid_annotations_df["exp"] != 7]

# Set parameters
BATCHSIZE = 128
NUM_EPOCHS = 20
LR = 4e-5
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


transform = transforms.Compose(
    [
        transforms.ElasticTransform(alpha=5.0, sigma=5.0),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.15, 0.15, 0.15),
        torchvision.transforms.RandomAutocontrast(p=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

transform_valid = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

train_dataset = CustomDataset(
    dataframe=train_annotations_df,
    root_dir=IMAGE_FOLDER,
    transform=transform,
    balance=False,
)
valid_dataset = CustomDataset(
    dataframe=valid_annotations_df,
    root_dir=IMAGE_FOLDER_TEST,
    transform=transform_valid,
    balance=False,
)
train_loader = DataLoader(
    train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=48
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

# Define (weighted) loss function
# weights = torch.tensor([0.015605, 0.008709, 0.046078, 0.083078, 0.185434, 0.305953, 0.046934, 0.30821])
weights7 = torch.tensor(
    [0.022600, 0.012589, 0.066464, 0.120094, 0.265305, 0.444943, 0.068006]
)
criterion = nn.CrossEntropyLoss(weights7.to(DEVICE))
criterion_val = (
    nn.CrossEntropyLoss()
)  # Use two loss functions, as the validation dataset is balanced


# Filter parameters for weight decay and no weight decay and create optimizer/scheduler
def filter_params(params, include_patterns, exclude_patterns):
    included_params = []
    excluded_params = []
    for name, param in params:
        if any(re.search(pattern, name) for pattern in include_patterns):
            included_params.append(param)
        elif not any(re.search(pattern, name) for pattern in exclude_patterns):
            excluded_params.append(param)
    return included_params, excluded_params


include_patterns = [
    r"^(?!.*\.bn)"
]  # Match any layer name that doesn't contain '.bn' = BatchNorm parameters
exclude_patterns = [r".*\.bn.*"]  # Vice versa
params_to_decay, params_not_to_decay = filter_params(
    MODEL.named_parameters(), include_patterns, exclude_patterns
)

# optimizer = optim.AdamW([
#    {'params': params_to_decay, 'weight_decay': ADAMW_WEIGHT_DECAY},  # Apply weight decay to these parameters
#    {'params': params_not_to_decay, 'weight_decay': 0.0}  # Exclude weight decay for these parameters = 0.0
# ], lr=LR)
optimizer = optim.AdamW(MODEL.parameters(), lr=LR)
lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=BATCHSIZE * NUM_EPOCHS)

# ***** Train the model *****
print("--- Start training ---")
scaler = torch.cuda.amp.GradScaler()
best_valid_loss = 100

for epoch in range(NUM_EPOCHS):
    MODEL.train()
    total_train_correct = 0
    total_train_samples = 0
    for images, labels in tqdm(train_loader, desc="Epoch train_loader progress"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = MODEL(images)
            loss = criterion(output.cuda(), labels.cuda())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

        _, train_predicted = torch.max(output, 1)
        total_train_samples += labels.size(0)
        total_train_correct += (train_predicted == labels).sum().item()

    train_accuracy = (total_train_correct / total_train_samples) * 100

    MODEL.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = MODEL(images)
            loss = criterion_val(outputs.cuda(), labels.cuda())
            valid_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] - "
        f"Validation Loss: {valid_loss/len(valid_loader):.4f}, "
        f"Validation Accuracy: {(correct/total)*100:.2f}%"
        f", Training Accuracy: {train_accuracy:.2f}%, "
    )
    # TBD: Valid loss überschreiben, dann model speichern wie unten, wenn kleiner als zuvor

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        print(f"Saving model at epoch {epoch+1}")
        torch.save(MODEL.state_dict(), "model.pt")  # Save the best model
