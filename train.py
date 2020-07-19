"""
Source: https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
Custom datasets, dataloaders and transforms: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import torch
import torchvision
from torchvision import transforms, utils
from torch.utils.data import DataLoader

from lenet5 import LeNet5

# Hyper parameters
NUM_EPOCHS = 15
NUM_CLASSES = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.01
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_PATH = '/home/aspirant/projects/torch/lenet/data/datasets/MNISTData'
MODEL_STORE_PATH = '/home/aspirant/projects/torch/lenet/data/models'

# Transforms to apply to the data
trans = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST(
    root=DATA_PATH, train=False, transform=trans)

# Create Dataloaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

net = LeNet5(NUM_CLASSES).to(DEVICE)

net.training_loop(train_loader, LEARNING_RATE, NUM_EPOCHS, DEVICE)

# Test the model
net.eval()
with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs, _ = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {}'.format(
        (correct / total) * 100))


net.save(MODEL_STORE_PATH)
