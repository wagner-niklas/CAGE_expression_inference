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
from torch.optim import lr_scheduler
import torchvision.transforms.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

# Load the annotations for training and validation from separate CSV files
IMAGE_FOLDER = "/data/AffectNet/train_set/images/"
IMAGE_FOLDER_TEST = "/data/AffectNet/val_set/images/"
train_annotations_path = "~/Spielwiese/affectnet/train_set_annotation_without_lnd.csv"
valid_annotations_path = "~/Spielwiese/affectnet/val_set_annotation_without_lnd.csv"
train_annotations_df = pd.read_csv(train_annotations_path)
valid_annotations_df = pd.read_csv(valid_annotations_path)

exp_counts_train = train_annotations_df['exp'].value_counts().sort_index() # Remove contempt for the AffectNet-7 version
exp_counts_valid = valid_annotations_df['exp'].value_counts().sort_index()

# Loop through the range from 0 to 6 and print the count for each value
for exp_value in range(8):
    train_count = exp_counts_train.get(exp_value, 0)
    valid_count = exp_counts_valid.get(exp_value, 0)
    print(f"exp {exp_value}: Train Count: {train_count}, Valid Count: {valid_count}")

# Set parameters
BATCHSIZE = 128
NUM_EPOCHS = 20
LR = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_mapping = {
    'Neutral': 0,
    'Happy': 1,
    'Sad': 2,
    'Suprise': 3,
    'Fear': 4,
    'Disgust': 5,
    'Anger': 6,
    'Contempt' :7,
}

# **** Create dataset and data loaders ****
class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, balance = False):
        self.dataframe = dataframe
        self.transform = transform
        self.root_dir = root_dir
        self.balance = balance

        if self.balance:
            self.dataframe = self.balance_dataset()
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, f"{self.dataframe['number'].iloc[idx]}.jpg")
        image = Image.open(image_path)

        classes = torch.tensor(self.dataframe.iloc[idx, 1], dtype=torch.int8)
        valence = torch.tensor(self.dataframe.iloc[idx, 2], dtype=torch.float16)
        arousal = torch.tensor(self.dataframe.iloc[idx, 3], dtype=torch.float16)

        if self.transform:
            image = self.transform(image)
        
        return image, classes, valence, arousal
    
    def balance_dataset(self):
        balanced_df = self.dataframe.groupby('exp', group_keys=False).apply(lambda x: x.sample(self.dataframe['exp'].value_counts().min()))
        return balanced_df

transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomGrayscale(0.01),
            transforms.RandomRotation(10), 
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # model more robust to changes in lighting conditions.
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5), # can be helpful if your images might have varying perspectives.
            transforms.ToTensor(),      # saves image as tensor (automatically divides by 255)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'), # Should help overfitting 
        ])

transform_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = CustomDataset(dataframe=train_annotations_df, root_dir=IMAGE_FOLDER,transform=transform, balance = True)
valid_dataset = CustomDataset(dataframe=valid_annotations_df, root_dir=IMAGE_FOLDER_TEST,transform=transform_valid, balance = False)
train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True,num_workers=48)
valid_loader = DataLoader(valid_dataset, batch_size=BATCHSIZE, shuffle=False,num_workers=48)

# ***** Define the model *****

# Initialize the model
MODEL = models.maxvit_t(weights='DEFAULT')
block_channels = MODEL.classifier[3].in_features
MODEL.classifier = nn.Sequential(
           nn.AdaptiveAvgPool2d(1),
           nn.Flatten(),
           nn.LayerNorm(block_channels),
           nn.Linear(block_channels, block_channels),
           nn.Tanh(),
           nn.Linear(block_channels, 10, bias=False),
       )
MODEL.to(DEVICE)
MODEL.load_state_dict(torch.load('best_best_model_affectnet_improved8VA.pt'))
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

def CCCLoss(x, y):
    # Compute means
    x_mean = torch.mean(x, dim=0)
    y_mean = torch.mean(y, dim=0)
    # Compute variances
    x_var = torch.var(x, dim=0)
    y_var = torch.var(y, dim=0)
    # Compute covariance matrix
    cov_matrix = torch.matmul((x - x_mean).permute(*torch.arange(x.dim() - 1, -1, -1)), y - y_mean) / (x.size(0) - 1)
    # Compute CCC
    numerator = 2 * cov_matrix
    denominator = x_var + y_var + torch.pow((x_mean - y_mean), 2)
    ccc = torch.mean(numerator / denominator)
    return -ccc

val_loss = nn.MSELoss()
aro_loss = nn.MSELoss()

optimizer = optim.AdamW(MODEL.parameters(), lr = LR)
lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = BATCHSIZE*NUM_EPOCHS)

# ***** Train the model *****
print("--- Start training ---")
scaler = torch.cuda.amp.GradScaler()
best_valid_loss= 100
l2_lambda = 0.00001  # L1 Regularization
l1_lambda = 0.00001  # L2 Regularization

for epoch in range(NUM_EPOCHS):
    MODEL.train()
    total_train_correct = 0
    total_train_samples = 0
    current_lr = optimizer.param_groups[0]['lr']
    for images, _, val_true, aro_true in tqdm(train_loader, desc ="Epoch train_loader progress"):
        images, val_true, aro_true = images.to(DEVICE), val_true.to(DEVICE), aro_true.to(DEVICE)
        optimizer.zero_grad()
        train_loss = 0
        l2_reg = 0
        l1_reg = 0
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = MODEL(images)
            val_pred = outputs[:, 0]
            aro_pred = outputs[:, 1]
            for param in MODEL.parameters():
                l2_reg += torch.norm(param, 2)#**2
                l1_reg += torch.norm(param, 1)                
            loss =  3 * val_loss(val_pred.cuda(), val_true.cuda()) \
                + 3 * aro_loss(aro_pred.cuda(), aro_true.cuda()) \
                + CCCLoss(val_pred.cuda(), val_true.cuda()) \
                + CCCLoss(aro_pred.cuda(), aro_true.cuda()) 
                #+ l2_lambda * l2_reg + l1_lambda * l1_reg
            train_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            #_, train_predicted = torch.max(class_pred, 1)
            #total_train_samples += class_true.size(0)
            #total_train_correct += (train_predicted == class_true).sum().item()
    
    #train_accuracy = (total_train_correct / total_train_samples) * 100
    
    MODEL.eval()
    valid_loss = 0.0
    total_valid_correct = 0
    total_valid_samples = 0
    with torch.no_grad():
        for images, _, val_true, aro_true in valid_loader:
            images, val_true, aro_true = images.to(DEVICE), val_true.to(DEVICE), aro_true.to(DEVICE)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = MODEL(images)
                val_pred = outputs[:, 0]
                aro_pred = outputs[:, 1]
                loss =  3 * val_loss(val_pred.cuda(), val_true.cuda()) \
                    + 3 * aro_loss(aro_pred.cuda(), aro_true.cuda()) \
                    + CCCLoss(val_pred.cuda(), val_true.cuda()) \
                    + CCCLoss(aro_pred.cuda(), aro_true.cuda())
                valid_loss += loss.item()
            #_, valid_predicted = torch.max(class_pred, 1)
            #total_valid_samples += class_true.size(0)
            #total_valid_correct += (valid_predicted == class_true).sum().item()
        
    #valid_accuracy = (total_valid_correct / total_valid_samples) * 100
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - "
          f"Training Loss: {train_loss/len(train_loader):.4f}, "
          f"Validation Loss: {valid_loss/len(valid_loader):.4f}, "
          #f"Training Accuracy: {train_accuracy:.2f}%, "
          #f"Validation Accuracy: {valid_accuracy:.2f}%, "
          f"Learning Rate: {current_lr:.8f}, ")

    if(valid_loss < best_valid_loss):
        best_valid_loss = valid_loss
        print(f"Saving model at epoch {epoch+1}")
        torch.save(MODEL.state_dict(), 'best_model_affectnet_improvedVA.pt') # Save the best model

# **** Test the model performance for classification ****

# Set the model to evaluation mode
MODEL.load_state_dict(torch.load('best_model_affectnet_improvedVA.pt'))
MODEL.to(DEVICE)
MODEL.eval()

all_val_true_values = []
all_val_predicted_values = []
all_aro_true_values = []
all_aro_predicted_values = []

# Start inference on test set
with torch.no_grad():
    for images, _, val_true, aro_true in valid_loader:
        images, val_true, aro_true = images.to(DEVICE), val_true.to(DEVICE), aro_true.to(DEVICE)

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

# Calculate regression metrics
def concordance_correlation_coefficient(true_values, predicted_values):
    mean_true = np.mean(true_values)
    mean_predicted = np.mean(predicted_values)

    numerator = 2 * np.cov(true_values, predicted_values)[0, 1]
    denominator = np.var(true_values) + np.var(predicted_values) + (mean_true - mean_predicted) ** 2

    return numerator / denominator

print(f'VAL Mean Squared Error (MSE): {mean_squared_error(all_val_true_values, all_val_predicted_values):.4f}')
print(f'VAL Mean Absolute Error (MAE): {mean_absolute_error(all_val_true_values, all_val_predicted_values):.4f}')
print(f'VAL Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(all_val_true_values, all_val_predicted_values)):.4f}')
print(f'VAL Concordance Correlation Coefficient (CCC): {concordance_correlation_coefficient(all_val_true_values, all_val_predicted_values):.4f}')
print(50*"*")
print(f'ARO Mean Squared Error (MSE): {mean_squared_error(all_aro_true_values, all_aro_predicted_values):.4f}')
print(f'ARO Mean Absolute Error (MAE): {mean_absolute_error(all_aro_true_values, all_aro_predicted_values):.4f}')
print(f'ARO Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(all_aro_true_values, all_aro_predicted_values)):.4f}')
print(f'ARO Concordance Correlation Coefficient (CCC): {concordance_correlation_coefficient(all_aro_true_values, all_aro_predicted_values):.4f}')
