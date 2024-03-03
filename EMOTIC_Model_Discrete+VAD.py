import pandas as pd
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torch.optim import lr_scheduler
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
import re
from tqdm import tqdm

# Set parameters
BATCHSIZE = 32
NUM_EPOCHS = 30
LR = 5e-5
MODEL = models.maxvit_t(weights='DEFAULT')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
       'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
       'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']

cat2ind = {}
ind2cat = {}
for idx, emotion in enumerate(cat):
  cat2ind[emotion] = idx
  ind2cat[idx] = emotion

# Load the annotations for training and validation from separate CSV files
class Emotic_PreDataset(Dataset):
  ''' Custom Emotic dataset class. Use preprocessed data stored in npy files. '''
  def __init__(self, x_body, y_cat, y_cont, transform, body_norm):
    super(Emotic_PreDataset,self).__init__()
    self.x_body = x_body
    self.y_cat = y_cat 
    self.y_cont = y_cont
    self.transform = transform 
    self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])           # Normalizing the body image with body mean and body std

  def __len__(self):
    return len(self.y_cat)
  
  def __getitem__(self, index):
    image_body = self.x_body[index] # Load the body information from the npy file
    cat_label = self.y_cat[index]
    cont_label = self.y_cont[index]
    return self.body_norm(self.transform(image_body)), torch.tensor(cat_label, dtype=torch.float32), (((torch.tensor(cont_label, dtype=torch.float32)*2)-11)/9)

path_to_emotic = "/data/Emotic/"
path_train = "emotic_pre/train.csv"
path_test = "emotic_pre/test.csv"
path_val = "emotic_pre/val.csv"
data_src = '.' # Directory where the emotic_pre folder is
train_annotations_df = pd.read_csv(path_train)
test_annotations_df = pd.read_csv(path_test)
valid_annotations_df = pd.read_csv(path_val)

label_counts = {}
print("Train")
for labels in train_annotations_df['Categorical_Labels']:
    # Convert the string representation of list to an actual list
    labels_list = eval(labels)
    # Iterate over each label in the list and count its occurrence
    for label in labels_list:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1


# Calc weights for the losses
def calc_weights(dict: dict) -> dict:
    s = 0
    for val in dict.values():
        s += val
    for key, val in dict.items():
        dict[key] = val #1 / (val / s)
    return dict

count_dict = calc_weights(label_counts)
l = []
for c in cat:
    l.append((23266-label_counts[c])/ label_counts[c])
print(l)
weights_train = torch.tensor(l)

# Iterate over the 'Categorical_Labels' column
label_counts = {}
print("Valid")
for labels in valid_annotations_df['Categorical_Labels']:
    # Convert the string representation of list to an actual list
    labels_list = eval(labels)
    # Iterate over each label in the list and count its occurrence
    for label in labels_list:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

count_dict = calc_weights(label_counts)
l = []
for c in cat:
    l.append((3315-label_counts[c])/ label_counts[c])
print(l)
weights_valid = torch.tensor(l)

# Load training preprocessed data
train_body = np.load(os.path.join(data_src,'emotic_pre','train_body_arr.npy'))
train_cat = np.load(os.path.join(data_src,'emotic_pre','train_cat_arr.npy'))
train_cont = np.load(os.path.join(data_src,'emotic_pre','train_cont_arr.npy'))
print(train_cont[0])

# Load validation preprocessed data 
val_body = np.load(os.path.join(data_src,'emotic_pre','val_body_arr.npy'))
val_cat = np.load(os.path.join(data_src,'emotic_pre','val_cat_arr.npy'))
val_cont = np.load(os.path.join(data_src,'emotic_pre','val_cont_arr.npy'))

# Load testing preprocessed data
test_body = np.load(os.path.join(data_src,'emotic_pre','test_body_arr.npy'))
test_cat = np.load(os.path.join(data_src,'emotic_pre','test_cat_arr.npy'))
test_cont = np.load(os.path.join(data_src,'emotic_pre','test_cont_arr.npy'))

body_mean = [0.43832874, 0.3964344, 0.3706214]
body_std = [0.24784276, 0.23621225, 0.2323653]
body_norm = [body_mean, body_std]

transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomGrayscale(0.01),
            transforms.RandomRotation(10), 
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # model more robust to changes in lighting conditions.
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5), # can be helpful if your images might have varying perspectives.
            transforms.ToTensor(),      # saves image as tensor (automatically divides by 255)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'), # TEST: Should help overfitting 
        ])

transform_valid =transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = Emotic_PreDataset( train_body, train_cat, train_cont, \
                                  transform,  body_norm)
val_dataset = Emotic_PreDataset( val_body, val_cat, val_cont, \
                                transform_valid,  body_norm)
test_dataset = Emotic_PreDataset( test_body, test_cat, test_cont, \
                                 transform_valid,  body_norm)

train_loader = DataLoader(train_dataset, BATCHSIZE, shuffle=True, drop_last=True, num_workers=48)
valid_loader = DataLoader(val_dataset, BATCHSIZE, shuffle=True, num_workers=48)
test_loader = DataLoader(test_dataset, BATCHSIZE, shuffle=False, num_workers=48) 
print(" ---- Data Loaders finished --- ")


# ***** Define the model *****
block_channels = MODEL.classifier[3].in_features
MODEL.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(block_channels),
            nn.Linear(block_channels, block_channels),
            nn.Tanh(),
            nn.Linear(block_channels, 29, bias=False),
        )

MODEL.to(DEVICE)
#MODEL.load_state_dict(torch.load('best_model_emotic.pt'))

# Define (weighted) loss function
criterion_cls = nn.BCEWithLogitsLoss(pos_weight=weights_train.to(DEVICE))#nn.MultiLabelSoftMarginLoss(weights_train.to(DEVICE), reduction='sum')
criterion_cls_val = nn.BCEWithLogitsLoss(pos_weight=weights_valid.to(DEVICE))#nn.MultiLabelSoftMarginLoss(weights_valid.to(DEVICE), reduction='sum')   # Use two loss functions, as the validation dataset is balanced
criterion_reg = nn.MSELoss()

optimizer = optim.AdamW(MODEL.parameters(), lr = LR)
lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = BATCHSIZE*NUM_EPOCHS)

# ***** Train the model *****
print("--- Start training ---")
scaler = torch.cuda.amp.GradScaler()
best_valid_loss= np.inf

for epoch in range(NUM_EPOCHS):
    MODEL.train()
    train_loss = 0.0
    for images, classes, labels in tqdm(train_loader, desc ="Epoch train_loader progress"):
        images, classes, labels = images.to(DEVICE), classes.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = MODEL(images)
            outputs_cls = outputs[:, :26]
            #print(outputs_cls[0].sigmoid())
            outputs_reg = outputs[:, 26:]
            loss = criterion_cls(outputs_cls.cuda(), classes.cuda()) + 5 * criterion_reg(outputs_reg.cuda() , labels.cuda())
            train_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

    MODEL.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, classes, labels in valid_loader:
            images, classes, labels = images.to(DEVICE), classes.to(DEVICE), labels.to(DEVICE)
            outputs = MODEL(images)
            outputs_cls = outputs[:, :26]
            outputs_reg = outputs[:, 26:]
            loss = criterion_cls_val(outputs_cls.cuda(), classes.cuda()) #+ 5 * criterion_reg(outputs_reg.cuda() , labels.cuda())
            valid_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - "
          f"Training Loss: {train_loss/len(train_loader):.4f}, "
          f"Validation Loss: {valid_loss/len(valid_loader):.4f}, ")

    if(valid_loss < best_valid_loss):
        best_valid_loss = valid_loss
        print(f"Saving model at epoch {epoch+1}")
        torch.save(MODEL.state_dict(), 'best_model_emotic.pt') # Save the best model