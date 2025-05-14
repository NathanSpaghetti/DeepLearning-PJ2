import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

def get_train_test_set(download = False, batch_size = 4):
    """
    :return: The train dataloader and test dataloader of CIFAR-10 recongnition issue
    """
    transform_train = transforms.Compose( [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        #*******
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
    transform_test = transforms.Compose( [
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=download, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

class MyModel(nn.Module):
    """
    Class for CIFAR-10 Recongnition
    """

    def __init__(self, kernel_size = 5,layer_list = [], channel_list = [5, 10], linear_list = [100]):
        super(MyModel, self).__init__()
        self.layers = []
        count_conv = 0 #count of convolution layer
        count_linear = 0 #count of full connection layer
        input_size = 32 #input size of linear layer, begin as height and width of CIFAR-10 image

        channel_list = [3] + channel_list
        for layer in layer_list:
            if layer == 'ReLU':
                self.layers.append(nn.ReLU())
            elif layer == 'MaxPool':
                self.layers.append(nn.MaxPool2d(kernel_size=2))
                input_size //= 2
            elif layer == 'AvgPool':
                self.layers.append(nn.AvgPool2d(kernel_size=2))
                input_size //= 2
            elif layer == 'Norm':
                self.layers.append(nn.LayerNorm([channel_list[count_conv], input_size, input_size]))
            elif layer == 'Drop':
                self.layers.append(nn.Dropout(0.1))
            elif layer == 'Conv':
                self.layers.append(nn.Conv2d(channel_list[count_conv], channel_list[count_conv + 1], kernel_size=kernel_size))
                #input_size -= (kernel_size - 1)
                #input_size += 2
                input_size -= (kernel_size - 1)
                count_conv += 1
            elif layer == 'Flatten':
                self.layers.append(nn.Flatten())
                input_size = input_size**2
                input_size *= channel_list[-1]
                linear_list = [input_size] + linear_list
            elif layer == 'Linear':
                self.layers.append(nn.Linear(linear_list[count_linear], linear_list[count_linear + 1]))
                count_linear += 1
            else:
                raise "No such layer!"
        self.layers = nn.ModuleList(self.layers)

    def b__init__(self, kernel_size = 5, channel_list = [5, 10],pool = 'Avg', linear_list = [100],act = 'ReLU'):
        super(MyModel, self).__init__()
        self.layers = []

        
        for i in range(len(channel_list) - 1):
            #Convolution
            self.layers.append(nn.Conv2d(channel_list[i], channel_list[i + 1], kernel_size))
            input_size -= (kernel_size - 1)
            #Activation function
            if act == 'ReLU':
                self.layers.append(nn.ReLU())
            #Pooling
            if pool == 'Avg':
                self.layers.append(nn.AvgPool2d(kernel_size=2))
                input_size //= 2
            elif pool == 'Max':
                self.layers.append(nn.MaxPool2d(kernel_size=2))
                input_size //= 2
            #Batch Norm
            self.layers.append(nn.LayerNorm([channel_list[i + 1], input_size, input_size]))
            #Dropout
            #self.layers.append(nn.Dropout(0.1))
        #Flatten
        self.layers.append(nn.Flatten())
        #self.layers.append(nn.ReLU())
        input_size = input_size**2
        input_size *= channel_list[-1]

        #Full connection
        linear_list = [input_size] + linear_list + [10]
        for i in range(len(linear_list) - 1):
            if act == 'ReLU':
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.1))
            self.layers.append(nn.Linear(linear_list[i], linear_list[i + 1]))

        self.layers = nn.ModuleList(self.layers)
    def a__init__(self, kernel_size = 5, channel_list = [5, 10],pool = 'Avg', linear_list = [100],act = 'ReLU'):
        super(MyModel, self).__init__()
        self.layers = []

        input_size = 32 #input size of linear layer, begin as height and width of CIFAR-10 image
        #Convolution layers
        channel_list = [3] + channel_list
        for i in range(len(channel_list) - 1):
            #Convolution
            self.layers.append(nn.Conv2d(channel_list[i], channel_list[i + 1], kernel_size))
            input_size -= (kernel_size - 1)
            #Batch Norm
            self.layers.append(nn.LayerNorm([channel_list[i + 1], input_size, input_size]))
            #Activation function
            #if act == 'ReLU':
            #    self.layers.append(nn.ReLU())
            #Pooling
            if pool == 'Avg':
                self.layers.append(nn.AvgPool2d(kernel_size=2))
                input_size //= 2
            elif pool == 'Max':
                self.layers.append(nn.MaxPool2d(kernel_size=2))
                input_size //= 2
            #Dropout
            #self.layers.append(nn.Dropout(0.1))
        #Flatten
        self.layers.append(nn.Flatten())
        #self.layers.append(nn.ReLU())
        input_size = input_size**2
        input_size *= channel_list[-1]

        #Full connection
        linear_list = [input_size] + linear_list + [10]
        for i in range(len(linear_list) - 1):
            if act == 'ReLU':
                self.layers.append(nn.ReLU())
            #self.layers.append(nn.Dropout(0.2))
            self.layers.append(nn.Linear(linear_list[i], linear_list[i + 1]))
        self.layers.append(nn.Dropout(0.3))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, input):

        """for layer in self.conv:
            input  = layer(input)

        input = self.flatten(input)
        for layer in self.linear:
            input = layer(input)"""
        
        for layer in self.layers:
            input = layer(input)

        return input
