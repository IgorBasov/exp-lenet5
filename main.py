"""
Source: https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
Custom datasets, dataloaders and transforms: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import torch
from datetime import datetime

from lenet5 import LeNet5
from lenet5_trainer import LeNet5Trainer
from net_test import NetTest
from utils import prepare_mnist_data

# Net hyper parameters
NUM_EPOCHS = 15
NUM_CLASSES = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.01
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Device is: {}'.format(DEVICE))

DATA_PATH = '/home/igor/Projects/exp-lenet5/data/datasets/MNISTData'
MODEL_STORE_PATH = '/home/igor/Projects/exp-lenet5/data/models'

train_loader, test_loader = prepare_mnist_data(DATA_PATH, BATCH_SIZE)
trainer = LeNet5Trainer(net=LeNet5(NUM_CLASSES).to(DEVICE), device=DEVICE)

trainingStartedAt = datetime.now()

net = trainer.train(
    train_data=train_loader,
    learning_rate=LEARNING_RATE,
    num_epochs=NUM_EPOCHS,
    print_every=BATCH_SIZE
)

print('Training took {}'.format((datetime.now() - trainingStartedAt)))

net_test = NetTest(net, DEVICE)

correct, total = net_test.test(test_data=test_loader)

print('Test Accuracy of the model on the 10000 test images: {}'.format(
        (correct / total) * 100))

# Save trained model
net.save(MODEL_STORE_PATH)
