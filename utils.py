import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


def prepare_mnist_data(data_path, batch_size):
    # Transforms to apply to the data
    trans = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root=data_path, train=True, transform=trans, download=True)
    test_dataset = torchvision.datasets.MNIST(
        root=data_path, train=False, transform=trans)

    # Create Dataloaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
