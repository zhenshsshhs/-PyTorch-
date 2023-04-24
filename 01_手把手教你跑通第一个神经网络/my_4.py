from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision import transforms
from torch import nn
import torch 
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

"train_dataset"
train_dataset = MNIST(root="./mnist_data",
                      train=True,
                      transform=transform,
                      target_transform=None)

"train_dataloader"
train_dataloader = DataLoader(train_dataset,
                              batch_size=100,
                              shuffle=True)

"model"

class SimpleModel(nn.Module):
    """Some Information about SimpleModel"""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=(1,1))
        self.conv2 = nn.Conv2d(in_channels=3,out_channels=5,kernel_size=(1,1))
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten(1,-1)
        self.linear = nn.Linear(in_features=5*28*28,out_features=10,bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu(x)
        return x

model = SimpleModel()


"optim"
def  eg_4_0():
    from torch import optim
    optimizer = optim.SGD(params=model.parameters(),lr=0.0001,momentum=0.9)
    print("optimizer.state_dict():{}".format(optimizer.state_dict()))

def  eg_4_1():
    from torch import optim
    params = [param for name, param in model.named_parameters() if ".bias" in name]
    # for name, param in model.named_parameters():
    #     print(param)
    #     break
    # print(params)
    optimizer = optim.SGD(params=params,lr=0.0001,momentum=0.9)
    print("optimizer.state_dict():{}".format(optimizer.state_dict()))

def  eg_4_2():
    from torch import optim
    from tqdm import tqdm

    optimizer = optim.SGD(params=model.parameters(),lr=0.0001,momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    epoch = 100
    for epoch_cur in range(epoch):
        with tqdm(train_dataloader,desc="EPOCH:{}".format(epoch_cur)) as train_bar:
            for (x,y) in train_bar:
                optimizer.zero_grad()
                loss = loss_fn(model(x),y)
                loss.backward()
                optimizer.step()
        print("epoch: {},  loss: {:.6f}".format(epoch, loss))


if __name__ == "__main__":
  """
  4.0 torch.optim
  4.1 params
  4.2 zero_grad(), step()
  """

#   eg_4_0()
#   eg_4_1()
  eg_4_2()

  print("~~~~~~下课~~~~~~")