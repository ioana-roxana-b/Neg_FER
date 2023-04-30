# Define the path to the dataset
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

# Define the target image size
img_size = (48, 48)

# Define the data transformations
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset and preprocess the images
dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)
classes = ['positive', 'negative']
# Manually assign the labels for positive and negative emotions
dataset.class_to_idx = {classes[i]: i for i in range(len(classes))}
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the VGG16 model
vgg16 = models.vgg16(pretrained=True)

# Remove the top layers
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1]
features.extend([nn.Linear(num_features, len(classes))])
vgg16.classifier = nn.Sequential(*features)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = vgg16(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics every 1000 iterations
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

print('Finished training')

# Evaluate the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = vgg16(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test accuracy: %d %%' % (100 * correct / total))



# Make predictions on the test set
y_pred = []
y_true = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = vgg16(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.tolist())
        y_true.extend(labels.tolist())

# Compute confusion matrix
conf_mat = confusion_matrix(y_true, y_pred)
print(conf_mat)
# Print confusion matrix
plt.imshow(conf_mat, cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

