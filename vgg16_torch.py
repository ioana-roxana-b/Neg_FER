import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


dataset_path = 'CK+48'
img_size = (224,224)
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)
classes = ['positive', 'negative']
dataset.class_to_idx = {classes[i]: i for i in range(len(classes))}
train_size = int(0.6 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)

vgg16 = models.vgg16(pretrained=True)
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-2]
features.extend([nn.Linear(num_features, 256),
                 nn.ReLU(inplace=True),
                 nn.Linear(256, len(classes))])
vgg16.classifier = nn.Sequential(*features)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

best_model_wts = None
best_loss = float('inf')
num_epochs = 20
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = vgg16(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            with torch.no_grad():
                val_loss = 0.0
                for val_inputs, val_labels in val_loader:
                    val_outputs = vgg16(val_inputs)
                    val_loss += criterion(val_outputs, val_labels).item() * val_inputs.size(0)
                val_loss /= len(val_dataset)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = vgg16.state_dict()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

    if epoch > 10 and val_loss > best_loss:
        break

vgg16.load_state_dict(best_model_wts)

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = vgg16(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test accuracy: %d %%' % (100 * correct / total))

y_pred = []
y_true = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = vgg16(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.tolist())
        y_true.extend(labels.tolist())

conf_mat = confusion_matrix(y_true, y_pred)
print(conf_mat)

plt.imshow(conf_mat, cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

