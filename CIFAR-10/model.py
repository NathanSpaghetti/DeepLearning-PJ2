import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

def get_train_test_set(download = False, batch_size = 4):
    """
    :return: The train dataloader and test dataloader of CIFAR-10 recongnition issue
    """
    transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

class MyModel(nn.Module):
    """
    Class for CIFAR-10 Recongnition
    """

    def __init__(self, kernel_size = 5, channel_list = [5, 10],pool = 'Avg', linear_list = [100],act = 'ReLU'):
        """
        :param channel_list: a list of the channel count in each convolution layer, there will be len(channel_list) convolution layer(s)
        :param pool: Pool function
        :param linear_lisr: a list of the size in each full connection layer.
        :param act: Activation function
        """
        super(MyModel, self).__init__()
        self.kernel_size = kernel_size
        input_size = 32 #input size of linear layer, begin as height and width of CIFAR-10 image

        #Convolution layers
        conv_list = [nn.Conv2d(3, channel_list[0], kernel_size)]
        input_size -= (kernel_size - 1)
        channel_list = channel_list
        for i in range(len(channel_list) - 1):
            #Pooling
            if pool == 'Avg':
                conv_list.append(nn.AvgPool2d(kernel_size=2))
                input_size //= 2
            elif pool == 'Max':
                conv_list.append(nn.MaxPool2d(kernel_size=2))
                input_size //= 2
            #conv_list.append(nn.Dropout(0.1))
            conv_list.append(nn.Conv2d(channel_list[i], channel_list[i + 1], kernel_size))
            input_size -= (kernel_size - 1)
            #*******
            conv_list.append(nn.AvgPool2d(kernel_size=2))
            input_size //= 2
        self.conv = nn.ModuleList(conv_list)

        #Flatten layer
        self.flatten = nn.Flatten()
        input_size = input_size**2
        input_size *= channel_list[-1]

        #full connection layer
        connect_list = [nn.ReLU(), nn.Linear(input_size, linear_list[0])]
        #connect_list = [ nn.Linear(input_size, linear_list[0])]
        linear_list = linear_list + [10] #10 as the numbers of output label

        for i in range(len(linear_list) - 1):
            if act == 'ReLU':
                connect_list.append(nn.ReLU())
                #connect_list.append(nn.Dropout(0.1))
            connect_list.append(nn.Linear(linear_list[i], linear_list[i + 1]))
            
        self.linear = nn.ModuleList(connect_list)

    def forward(self, input):

        for layer in self.conv:
            input  = layer(input)

        input = self.flatten(input)
        for layer in self.linear:
            input = layer(input)

        return input
