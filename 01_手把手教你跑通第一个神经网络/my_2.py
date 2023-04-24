import torch
from torchvision.datasets import mnist
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = mnist.MNIST(root="./mnist_data",
                            train=True,
                            download=False,
                            transform=transform,
                            target_transform=None,
                            )


def eg_2_1():
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        train_dataset, batch_size=10000, shuffle=False)

    from collections.abc import Iterable
    print("isinstance(train_dataset,Iterable):{}".format(
        isinstance(train_dataset, Iterable)))
    print("isinstance(train_dataloader,Iterable):{}".format(
        isinstance(train_dataloader, Iterable)))

    print("type(train_dataloader):{}".format(type(train_dataloader)))
    for batch in train_dataloader:
        print("type(batch):{}".format(type(batch)))
        print("len(batch):{}".format(len(batch)))
        print("type(batch[0]):{}".format(type(batch[0])))
        print("type(batch[1]):{}".format(type(batch[1])))
        print("batch[0].shape:{}".format(batch[0].shape))
        print("batch[1].shape:{}".format(batch[1].shape))

        print(batch[0][0].shape)
        print("=================")


def eg_2_2():

    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        train_dataset, batch_size=10000, shuffle=True)
    print("len(train_dataloader):{}".format(len(train_dataloader)))
    print("len(train_dataloader.dataset):{}".format(
        len(train_dataloader.dataset)))


def eg_2_3_0():
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        train_dataset, batch_size=10000, shuffle=False)

    for batch, (x, y) in enumerate(train_dataloader):
        print("batch:{},type(x):{},type(y):{}".format(batch, type(x), type(y)))
        # batch: 0, type(x): <class 'torch.Tensor'>, type(y): <class 'torch.Tensor'>
        print("batch:{},type(x):{},type(y):{}".format(batch, x.shape, y.shape))
        # batch: 0, x.shape: torch.Size([10000, 1, 28, 28]), y.shape: torch.Size([10000])
        break
def enumate1():
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    for batch, (x) in enumerate(seasons):
        print("batch:{},type(x):{},type(y):{}".format(batch, type(x), type(x)))


def eg_2_3_1():
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=10000, shuffle=False)

    from tqdm import tqdm
    with tqdm(train_dataloader, desc="training") as train_bar:

        for (x,y) in train_bar:
            pass

            
def eg_2_4():
    def collate_fn(batch):
        print("type(batch): {}".format(type(batch)))  # <class 'list'>
        print("len(batch): {}".format(len(batch)))  # 10000
        print("type(batch[0]): {}".format(type(batch[0])))  # <class 'tuple'>
        x = [i[0] for i in batch]
        y = [i[1] for i in batch]
        x = torch.cat(x)[:,None,...]    # ... 是多个：的意思， None 在对应位置加一维空值
        y = torch.Tensor(y)
        return {"x":x, "y":y}


    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=10000,
                            shuffle=False,
                            collate_fn=collate_fn)

    for batch in train_loader:
        print("type(batch):{}".format(type(batch)))
        print("type(batch[\"x\"]):{}".format(type(batch["x"])))
        print("type(batch[\"y\"]):{}".format(type(batch["y"])))
        print("batch[\"x\"].shape:{}".format(batch["x"].shape))
        print("batch[\"y\"].shape:{}".format(batch["y"].shape))
def torch_index():
    a = torch.tensor([1,5,2,3])
    print(a[None,None,None,:])


if __name__ == "__main__":
    """
    2.0 torch.utils.data.DataLoader https://pytorch.org/docs/stable/data.html
    2.1 __iter__  [magic method]
    2.2 __len__  [magic method]
    2.3.0 enumerate
    2.3.1 tqdm
    2.4 collate_fn
    """
    # eg_2_1()
    # eg_2_2()
    # eg_2_3_0()
    # eg_2_3_1()
    eg_2_4()
    torch_index()

    # enumate1()
    print("~~~~~~下课~~~~~~")
