import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def get_data_loader(training = True):
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    train_set=datasets.FashionMNIST('./data',train=True,
            download=True,transform=custom_transform)
    
    test_set=datasets.FashionMNIST('./data', train=False,
            transform=custom_transform)
    
    if training == True:
        loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
        return loader
    else:
        loader = torch.utils.data.DataLoader(test_set, batch_size = 64)
        return loader


def build_model():
    model = nn.Sequential(nn.Flatten(),
                          nn.Linear(784, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, 10))
    return model


def train_model(model, train_loader, criterion, T):
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    
    for epoch in range(T):
        
        total_loss = 0.0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
        
        print(f'Train Epoch: {epoch} Accuracy: {correct}/60000({100 * correct / 60000:.2f}%) Loss: {total_loss / 60000:.3f}')
            


def evaluate_model(model, test_loader, criterion, show_loss = True):
    model.eval()
    
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
    
    if show_loss == True:
        print(f'Average loss: {(total_loss/total):.4f}')
        print(f'Accuracy: {(100 * correct/total):.2f}%')
    else:
        print(f'Accuracy: {(100 * correct/total):.2f}%')


def predict_label(model, test_images, index):
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt', 'Sneaker','Bag','Ankle Boot']  
    outputs = model(test_images[index])
    prob = F.softmax(outputs, dim=1)
    highest, indicies = torch.topk(prob, 3)
    
    for i in range(len(indicies[0])):
        accuracy = highest[0][i].item()*100
        print(f'{class_names[indicies[0][i]]}: {accuracy:.2f}')

if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)
    test_loader = get_data_loader(False)
    model = build_model()
    print(model)
    train_model(model, train_loader, criterion, 5)
    evaluate_model(model, test_loader, criterion, show_loss = False)
    evaluate_model(model, test_loader, criterion, show_loss = True)
    pred_set = []
    for images, labels in test_loader:
        pred_set.append(images)
    predict_label(model, pred_set, 1)