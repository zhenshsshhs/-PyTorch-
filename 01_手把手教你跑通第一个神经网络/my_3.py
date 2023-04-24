import os
import torch

from torchvision.datasets.mnist import MNIST
from torch.utils.data import Dataset, DataLoader, dataset
from torchvision import transforms, models
from torch import nn
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = MNIST(root="./mnist_data",
                      train=True,
                      transform=transform,
                      target_transform=None)
train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True)


def eg_3_0_0():
    class SimpleModel(nn.Module):
        """Some Information about SimpleModel"""

        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv1 = nn.Conv2d(
                in_channels=1, out_channels=3, kernel_size=(1, 1))
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            return x
    model = SimpleModel()
    print("model:{}".format(model))
    for name, param in model.named_parameters():
        print(name, param)


def eg_3_0_1():
    class SimpleModel(nn.Module):
        """Some Information about SimpleModel"""

        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv1 = nn.Conv2d(
                in_channels=1, out_channels=3, kernel_size=(1, 1))
            self.conv2 = nn.Conv2d(
                in_channels=3, out_channels=5, kernel_size=(1, 1))
            self.relu = nn.ReLU(inplace=True)
            self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
            self.linear = nn.Linear(
                in_features=5*28*28, out_features=10, bias=False)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            print("[before flatten] x.shape: {}".format(
                x.shape))  # torch.Size([1, 5, 28, 28])
            x = self.flatten(x)
            print("[after flatten] x.shape: {}".format(
                x.shape))  # torch.Size([1, 3920])
            x = self.linear(x)
            x = self.relu(x)
            return x
    model = SimpleModel()
    x = train_dataset[0][0]  # torch.Size([1, 28, 28])
    x = x[None, ...]  # torch.Size([1, 1, 28, 28])
    y = model(x)
    print(y)


def eg_3_1():
    class SimpleModel(nn.Module):
        """Some Information about SimpleModel"""

        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv1 = nn.Conv2d(
                in_channels=1, out_channels=3, kernel_size=(1, 1))
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            return x
    model = SimpleModel()
    x = train_dataset[0][0]
    x = x[None, ...]
    print("model(x) == model.__call__(x):{}".
          format(True if torch.all(model(x) == model.__call__(x)) else False))
    print("model(x) == model.forward(x):{}".format(True if torch.all(model(x) == model.forward(x))else False))


def eg_3_2():
    pass


def eg_3_3():
    from torchvision import models

    models_vgg16 = models.vgg16()
    print("model_vgg16:{}".format(models_vgg16))

    models_resnet50 = models.resnet50()
    print("model_resnet50:{}".format(models_resnet50))


def eg_3_4_0():
    from torchvision import models

    model_vgg16 = models.vgg16()
    print("model_vgg16.state_dict():{}".format(model_vgg16.state_dict()))
    print("type(model_vgg16.state_dict()）: {}".format(type(model_vgg16.state_dict())))


def eg_3_4_1():
    from torchvision import models

    model_vgg16 = models.vgg16()
    torch.save(model_vgg16.state_dict(),"./vgg16.pth")



def eg_3_4_2():
    from torchvision import models

    model_vgg16 = models.vgg16()
    state_dict = torch.load("./vgg16.pth",map_location="cpu")
    missing_keys, unexpected_keys = model_vgg16.load_state_dict(state_dict,strict=True)
    print("missing_keys: {}".format(missing_keys))
    print("unexpected_keys: {}".format(unexpected_keys))


def eg_3_4_3():
    from torchvision import models

    model_vgg16 = models.vgg16()
    state_dict = torch.load("./vgg16.pth",map_location="cpu")
    for key in list(state_dict.keys()):
        if ".bias" in key:
            del state_dict[key]
        

    missing_keys, unexpected_keys = model_vgg16.load_state_dict(state_dict,strict=False)
    print("missing_keys: {}".format(missing_keys))
    print("unexpected_keys: {}".format(unexpected_keys))


def eg_3_5():
    from torch.utils import model_zoo
    from torchvision import models

    model_alexnet = models.alexnet()
    state_dict = model_zoo.load_url('http://download.pytorch.org/models/alexnet-owt-7be5be79.pth')
    model_alexnet.load_state_dict(state_dict)

if __name__ == "__main__":
    """
    3.0.0 torch.nn.Module
    3.0.1 super().__init__()
    3.1 __call__  [magic method]
    3.2 (B, C, H ,W)
    3.3 torchvison.models
    3.4.0 model.dict_state()
    3.4.1 torch.save(model.dict_state(), f)
    3.4.2 model.load_state_dict()
    3.4.3 strict=False
    3.5 torch.utils.model_zoo.load_url()
    """

    # eg_3_0_0()
    # eg_3_0_1()
    # eg_3_1()
    # eg_3_2()
    # eg_3_3()
    # eg_3_4_0()
    # eg_3_4_1()
    # eg_3_4_2()
    # eg_3_4_3()
    eg_3_5()

    print("~~~~~~下课~~~~~~")
