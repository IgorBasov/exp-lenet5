"""
Source: https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
Custom datasets, dataloaders and transforms: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import torch

from lenet5 import LeNet5
from lenet5trainer import LeNet5Trainer
from utils import prepare_mnist_data

# Hyper parameters
NUM_EPOCHS = 15
NUM_CLASSES = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.01
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_PATH = '/home/aspirant/projects/torch/lenet/data/datasets/MNISTData'
MODEL_STORE_PATH = '/home/aspirant/projects/torch/lenet/data/models'

net = LeNet5(NUM_CLASSES).to(DEVICE)
trainer = LeNet5Trainer(net, DEVICE)

train_loader, test_loader = prepare_mnist_data(DATA_PATH, BATCH_SIZE)

trainer\
    .train(train_loader, LEARNING_RATE, NUM_EPOCHS, print_every=BATCH_SIZE)\
    .save(MODEL_STORE_PATH)

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
