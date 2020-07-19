# LeNet5 (PyTorch): MNIST dataset

Another attempt to implement Yann LeCun's [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) network using [PyTorch](https://pytorch.org/) and test it in [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database).

## Run experiment

1. Install *torch* and *torchvision* libs

2. Run

```bash
python main.py
``` 

## Description

### Files
- `lenet5.py` - contains *LeNet5* class that implements LeNet-5 network
- `lenet5_trainer.py` - contains *LeNet5Trainer* class. Method *train(train_data, learning_rate, num_epochs, print_every=100)* trains the network and returns it after.
- `net_test.py` - contains *NetTest* class. Method *test(test_data)* performs testing of the network and returns quantity of correct tests and total quantity of tests.
- `utils.py` - contains method *prepare_mnist_data(data_path, batch_size)* that prepares data for the experiment.
- `main.py` - main file of the experiment.

### Folders
- `data/datasets/` - MNIST-dataset will be downloaded here (at first run).
- `data/models/` - folder to save files of trained models.

# Remarks
- I'm new in Python, so be lenient ;)
- The network implementation is _very rough_, so be _very lenient_ ;) 