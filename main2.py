import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import pandas as pd
import numpy as np

# Load metadata
metadata_path = "HAM10000_metadata.csv"
image_dir = "dataset/"

df = pd.read_csv(metadata_path)

# Ensure images are correctly linked
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['dx'])

# Train-validation split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Define Transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

# Define Dataset Class
class HAM10000Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['image_path']
        image = Image.open(image_path).convert("RGB")
        label = self.df.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        # Ensure the label is of type long (torch.int64)
        return image, torch.tensor(label, dtype=torch.long)


# Create Dataset and DataLoader
train_dataset = HAM10000Dataset(train_df, transform=data_transforms['train'])
val_dataset = HAM10000Dataset(val_df, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load Pretrained Model (ResNet18)
model = models.resnet18(pretrained=True)

# Modify the final layer for classification
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(label_encoder.classes_))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(device)
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    return model

# Train the Model
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# Save the Trained Model
torch.save(model.state_dict(), "skin_cancer_model.pth")

# Load and Predict
def predict(model, image_path):
    model.eval()
    transform = data_transforms['val']
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
    return label_encoder.inverse_transform([pred.item()])[0]

# Test Prediction
test_image_path = val_df.iloc[0]['image_path']
predicted_label = predict(model, test_image_path)
print(f"Predicted Label: {predicted_label}, True Label: {val_df.iloc[0]['dx']}")