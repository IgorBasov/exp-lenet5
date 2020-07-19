import torch


class NetTest:
    def __init__(self, net, device):
        self.net = net
        self.device = device

    def test(self, test_data):
        correct = 0
        total = 0

        self.net.eval()

        with torch.no_grad():
            for items, labels in test_data:
                items = items.to(self.device)
                labels = labels.to(self.device)

                outputs, _ = self.net(items)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct, total
