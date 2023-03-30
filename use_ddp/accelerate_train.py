"""(SNSC) Single Node Single GPU Card Training"""
import os
from accelerate import Accelerator, notebook_launcher
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

BATCH_SIZE = 256
EPOCHS = 5


def train_ddp():
    # 原来是要包装成整个函数, 我前面是在 __name__ == "__main__" 里面写的, 会一直卡着
    # 看文档说是要自己初始化
    # https://github.com/huggingface/accelerate/issues/141
    if os.name == "nt":
        # windows 还是没人权
        dist.init_process_group(backend="gloo", init_method="tcp://localhost:23456", rank=0, world_size=1)
    accelerator = Accelerator()

    # 1. define network
    net = torchvision.models.resnet18(num_classes=10)

    # 2. define dataloader
    trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 3. define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,
    )

    net, optimizer, train_loader = accelerator.prepare(net, optimizer, train_loader)

    print("            =======  Training  ======= \n")

    # 4. start to train
    net.train()
    for ep in range(1, EPOCHS + 1):
        train_loss = correct = total = 0

        # 卡在这里了, 不会动
        for idx, (inputs, targets) in enumerate(tqdm(train_loader, disable=not accelerator.is_local_main_process)):
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)
            correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()

            if (idx + 1) % 50 == 0 or (idx + 1) == len(train_loader):
                print(
                    "   == step: [{:3}/{}] [{}/{}] | loss: {:.3f} | acc: {:6.3f}%".format(
                        idx + 1,
                        len(train_loader),
                        ep,
                        EPOCHS,
                        train_loss / (idx + 1),
                        100.0 * correct / total,
                    )
                )

    print("\n            =======  Training Finished  ======= \n")


if __name__ == "__main__":
    # train_ddp()
    notebook_launcher(train_ddp, args=(), num_processes=1)


"""
accelerate config --config_file accelerate_config.yaml
accelerate launch --config_file accelerate_config.yaml accelerate_train.py

现在在 windows 上直接运行也可以了
python accelerate_train.py
"""
