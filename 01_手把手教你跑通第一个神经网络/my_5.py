from datetime import datetime
import torch 
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

device = (
    "cuda:0"
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

train_dataset = MNIST(root="./mnist_data",train=True,transform=transform,target_transform=None)

train_dataloader = DataLoader(train_dataset,batch_size=100,shuffle=True)


class SimpleModel(nn.Module):
  def __init__(self):
      super(SimpleModel, self).__init__()
      self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 1))
      self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(1, 1))
      self.relu = nn.ReLU(inplace=True)
      self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
      self.linear = nn.Linear(in_features=5*28*28, out_features=10, bias=False)

  def forward(self, x):
      x = self.conv1(x)
      x = self.relu(x)
      x = self.conv2(x)
      x = self.relu(x)
      x = self.flatten(x)
      x = self.linear(x)
      x = self.relu(x)
      return x
# model
model = SimpleModel()
# model.load_state_dict("./model_2023_4_24.pth")

# optim
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4,momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()
epoch = 100
for epoch_cur in range(epoch):
    with tqdm(train_dataloader,desc="EPOCH:{}".format(epoch_cur)) as train_bar:
        for (x,y) in train_bar:
            optimizer.zero_grad()
            loss = loss_fn(model(x),y)
            loss.backward()
            optimizer.step()
    print("epoch: {}, loss:{:.6f}".format(epoch_cur,loss))     

time = str(datetime.now()).split(" ")[0].replace("-", "_")
torch.save(model.state_dict,"model_{}.pth".format(time))

print("~~~~~~撒花~~~~~~")