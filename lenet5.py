"""
https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
Categorical Cross Entropy (TF) vs. CrossEntropy Loss (PyTorch) - https://discuss.pytorch.org/t/categorical-cross-entropy-loss-function-equivalent-in-pytorch/85165/4
LeNet5 PyTorch: https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
https://github.com/erykml/medium_articles/blob/master/Computer%20Vision/lenet5_pytorch.ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, classes_qty):
        super(LeNet5, self).__init__()

        # Feature extraction layers (convolution part)
        self.feature_extractor = nn.Sequential(
            # 1 input image channel, 6 output channels, 5*5 - kernel size
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            # 2*2 - pool size, 1 - stride
            nn.AvgPool2d(kernel_size=2),

            # 6 - input channels, 120 - output channels, 5*5 - kernel size
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            # 2*2 - pool size, 2 - stride
            nn.AvgPool2d(kernel_size=2, stride=2),

            # 16 - input channels, 120 - output channels, 5*5 - kernel size
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=classes_qty),
        )

    def forward(self, x):
        out = self.feature_extractor(x)
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    def save(self, dir_path, file_name='lenet5.ckpt'):
        torch.save(self.state_dict(), dir_path + '/' + file_name)
