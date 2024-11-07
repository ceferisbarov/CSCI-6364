from sklearn.datasets import fetch_openml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


batch_size = 64
num_epochs = 30
device = "cuda"

np.random.seed(1)

mnist = fetch_openml('mnist_784')

X = mnist.data.to_numpy().reshape(-1,28,28)
Y = mnist.target.to_numpy()
shuffle = np.random.permutation(70000)
Xtrain, Xtest = X[shuffle[:60000]], X[shuffle[60000:]]
Ytrain, Ytest = Y[shuffle[:60000]], Y[shuffle[60000:]]

class ConvNet6(nn.Module):
    def __init__(self):
        super(ConvNet6, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(256 * 1 * 1, 256)
        self.fc2 = nn.Linear(256, 10)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.pool = nn.AvgPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))   # Normalize to [-1, 1]
])


class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

train_dataset = MNISTDataset(Xtrain, Ytrain)
test_dataset = MNISTDataset(Xtest, Ytest)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = ConvNet6()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", unit="batch")
    
    for images, labels in train_loader_tqdm:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        train_loader_tqdm.set_postfix(loss=loss.item())
    
    avg_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
    
    model.eval()
    correct_test = 0
    total_test = 0
    test_loss = 0.0
    test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Testing]", unit="batch")
    
    with torch.no_grad():
        for images, labels in test_loader_tqdm:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
