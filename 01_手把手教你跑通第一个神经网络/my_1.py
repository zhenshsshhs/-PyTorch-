import os
import torch
from torch.utils.data import Dataset


def eg_1_1():
    x = torch.linspace(-1, 1, 10)
    y = x**2

    class SimpleDataset(Dataset):
        def __init__(self, x, y):
            super().__init__()
            self.x = x
            self.y = y

        def __getitem__(self, index):
            return {"x": self.x[index], "y": self.y[index]}

        def __len__(self):
            return len(self.x)

    simple_dataset = SimpleDataset(x, y)
    index = 0

    # __getitem__
    print("simpledataset.__getitem__({}):{}".format(
        index, simple_dataset.__getitem__(index)))
    print("simpledataset[{}]:{}".format(index, simple_dataset[index]))

    # __len__
    print("simpledataset.__len__:{}".format(simple_dataset.__len__()))
    print("len(simpledataset):{}".format(len(simple_dataset)))


def eg_1_2_0():
    from torchvision.datasets import mnist
    train_dataset = mnist.MNIST(root="./mnist_data",
                                train=True,
                                download=True,
                                transform=None)
    print("type(train_dataset):{}".format(type(train_dataset)))
    index = 0
    print("train_dataset[{}]:{}".format(index, train_dataset[index]))

    import matplotlib.pyplot as plt

    plt.imshow(train_dataset[index][0], cmap="gray")
    print("len(train_dataset):{}".format(len(train_dataset)))
    plt.savefig('./01_手把手教你跑通第一个神经网络/first_picture')


def eg_1_2_1():
    from torchvision.datasets import mnist
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_dataset = mnist.MNIST(
        root='./mnist_data',
        train=True,
        download=False,
        transform=transform,
        target_transform=None
    )

    index = 0
    print("type(train_dataset[{}]): {}".format(
        index, type(train_dataset[index])))  # <class 'tuple'>
    print("type(train_dataset[{}][0]): {}".format(
        index, type(train_dataset[index][0])))  # <class 'torch.Tensor'>
    print("train_dataset[{}][0].shape:{}".format(
        index, train_dataset[index][0].shape))
    print("type(train_dataset[{}][1]):{}".format(
        index, type(train_dataset[index][1])))


def eg_1_3():
    from torchvision.datasets.voc import VOCSegmentation, VOCDetection

    segment_dataset = VOCSegmentation(root="./voc_data",
                                      year="2012",
                                      image_set="train",
                                      transform=None,
                                      download=True)

    detection_dataset = VOCDetection(root="./voc_data",
                                     year="2012",
                                     image_set="train",
                                     transform=None,
                                     download=True
                                     )
    index = 0
    print("type(segment_dataset[{}]): {}".format(
        index, type(segment_dataset[index])))  # <class 'tuple'>
    
    print("type(segment_dadetection_datasettaset[{}][0]): {}".format(
        index, type(segment_dataset[index][0])))  # <class 'torch.Tensor'>
    
    print("type(segment_dataset[{}][1]):{}".format(
        index, type(segment_dataset[index][1])))


    print("type(detection_dataset[{}]): {}".format(
        index, type(detection_dataset[index])))  # <class 'tuple'>
    
    print("type(detection_dataset[{}][0]): {}".format(
        index, type(detection_dataset[index][0])))  # <class 'torch.Tensor'>
    
    print("type(detection_dataset[{}][1]):{}".format(
        index, type(detection_dataset[index][1])))
def eg_1_4_0():
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
    ])
    
    train_dataset = ImageFolder(root=os.path.join("./flowers","train"),
                                transform=transform,
                                target_transform=None
                                )
    index = 0
    print("type(train_dataset[{}]):{}".format(index, type(train_dataset[index])))

    print("type(train_dataset[{}][0]):{}".format(index, type( train_dataset[index][0])))
    print("train_dataset[{}][0].shape:{}".format(index, train_dataset[index][0].shape))

    print("type(train_dataset[{}][1]):{}".format(index, type(train_dataset[index][1])))
    print("train_dataset[{}][1]:{}".format(index, train_dataset[index][1]))



def eg_1_4_1():
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    
    train_dataset = ImageFolder(os.path.join("./flowers","train"),
                          transform=transform,
                          target_transform=None
                          )
    print("train_dataset.classes: {}".format(train_dataset.classes))  # ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    print("train_dataset.class_to_idx: {}".format(train_dataset.class_to_idx))  # {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}


if __name__ == "__main__":
    """
    1.0 torch.utils.data.Dataset
    1.1 __getitem__, __len__  [magic methods]
    1.2.0 MNIST
    1.2.1 transforms
    1.3 VOCSegmentation, VOCDetection
    1.4.0 ImageFolder
    1.4.1 classes, class_to_idx
    """
    # eg_1_1()
    # eg_1_2_0()
    # eg_1_3()
    # eg_1_4_0()
    eg_1_4_1()
    print("~~~~~~下课~~~~~~")
