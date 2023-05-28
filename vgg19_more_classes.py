import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import VGG19_Weights


def vgg19():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = 'CK+48 - Copy'
    img_size = (224, 224)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)
    classes = ['contempt', 'happy', 'surprise', 'anger', 'disgust', 'fear', 'sadness']
    dataset.class_to_idx = {classes[i]: i for i in range(len(classes))}
    train_size = int(0.6 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
    vgg19 = vgg19.to(device)
    num_features = vgg19.classifier[6].in_features
    features = list(vgg19.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, len(classes))])
    vgg19.classifier = nn.Sequential(*features)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg19.parameters(), lr=0.003, momentum=0.9, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = vgg19.state_dict()
    best_loss = float('inf')
    num_epochs = 50
    no_improvement_epochs = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = vgg19(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(100 * correct_preds / total_preds)

        with torch.no_grad():
            val_loss = 0.0
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = vgg19(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item() * val_inputs.size(0)
            val_loss /= len(val_dataset)
            val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = vgg19.state_dict()
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        scheduler.step()

    vgg19.load_state_dict(best_model_wts)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = vgg19(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test accuracy: %d %%' % (100 * correct / total))

    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = vgg19(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.tolist())
            y_true.extend(labels.tolist())

    conf_mat = confusion_matrix(y_true, y_pred)
    print(conf_mat)

    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    plt.figure()
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.axhline(y=100 * correct / total, color='r', linestyle='-', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Test Accuracy')
    plt.show()
