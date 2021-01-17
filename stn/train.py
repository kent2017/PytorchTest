import torch
import torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


class STN2D(nn.Module):
    def __init__(self, input_channels=1):
        super(STN2D, self).__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=7),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10*3*3, 32),
            nn.ReLU(),
            nn.Linear(32, 3*2)
        )

        self.init_weights()

    def init_weights(self):
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        batch_size = x.shape[0]
        xs = self.localization(x).view(batch_size, -1).contiguous()
        theta = self.fc_loc(xs).view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


class Net(nn.Module):
    def __init__(self, input_channels):
        super(Net, self).__init__()

        self.stn = STN2D(input_channels)

        self.nn = nn.Sequential(
            nn.Conv2d(input_channels, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.MaxPool2d(2, 2),
        )
        self.mlp = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.stn(x)
        x = self.nn(x).view(-1, 320).contiguous()
        x = self.mlp(x)
        return x


if __name__ == "__main__":
    import platform
    import matplotlib as mpl
    # set backend
    if platform.system() == 'Windows':
        mpl.use('TkAgg')
    elif platform.system() == 'Linux':
        pass
    else:
        mpl.use('macosx')

    plt.ion()       # interactive mode
    f, axarr = plt.subplots(1, 2)
    axarr[0].set_title('Dataset Images')
    axarr[1].set_title('Transformed Images')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(1).to(device)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='../datasets', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=64, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='../datasets', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081))
        ])), batch_size=64, num_workers=4)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx%500 == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()
                ))

    def test():
        with torch.no_grad():
            model.eval()
            test_loss = 0
            correct = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                # sum up batch loss
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
                  .format(test_loss, correct, len(test_loader.dataset),
                          100. * correct / len(test_loader.dataset)))

            visualize_stn()

    def convert_image_np(inp):
        """Convert a Tensor to numpy image."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp

    def visualize_stn():
        with torch.no_grad():
            # Get a batch of training data
            data = next(iter(test_loader))[0].to(device)

            input_tensor = data.cpu()
            transformed_input_tensor = model.stn(data).cpu()

            in_grid = convert_image_np(
                torchvision.utils.make_grid(input_tensor))

            out_grid = convert_image_np(
                torchvision.utils.make_grid(transformed_input_tensor))

            # Plot the results side-by-side
            axarr[0].imshow(in_grid)
            axarr[1].imshow(out_grid)
            plt.pause(0.1)


    for epoch in range(1, 20 + 1):
        train(epoch)
        test()

    # Visualize the STN transformation on some input batch
    visualize_stn()

    plt.ioff()
    plt.show()