import torch
import torch.nn as nn


class LeNet5Trainer:
    def __init__(self, net, device):
        self.device = device
        self.net = net

    def train(self, train_data, learning_rate, num_epochs, print_every=100):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)
        total_step = len(train_data)
        loss_list = []
        acc_list = []

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_data):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs, _ = self.net(images)
                loss = criterion(outputs, labels)
                loss_list.append(loss.item())

                # Backpropagation and perform SGD optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the accuracy
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                acc_list.append(correct / total)

                if print_every != 0 and (i + 1) % print_every == 0:
                    print(
                        'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                            .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100)
                    )

        return self.net
