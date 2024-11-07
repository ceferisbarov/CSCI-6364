from sklearn.datasets import fetch_openml
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


np.random.seed(1)

mnist = fetch_openml('mnist_784')

X = mnist.data.to_numpy().reshape(-1,28,28)
Y = mnist.target.to_numpy()
shuffle = np.random.permutation(70000)
Xtrain, Xtest = X[shuffle[:60000]], X[shuffle[60000:]]
Ytrain, Ytest = Y[shuffle[:60000]], Y[shuffle[60000:]]

class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = FullyConnectedNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", unit="batch")
    
    for images, labels in train_loader_tqdm:
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
