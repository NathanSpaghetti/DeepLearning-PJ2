import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

def get_train_test_set(download = False, batch_size = 4):
    """
    :return: The train dataloader and test dataloader of CIFAR-10 recongnition issue
    """
    transform_train = transforms.Compose( [
        #Preprocess
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
    transform_test = transforms.Compose( [
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root='../data/', train=True, download=download, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='../data/', train=False, download=download, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)

    return trainloader, testloader

class MyModel(nn.Module):
    """
    Class for CIFAR-10 Recongnition
    """

    def __init__(self, kernel_size = 5,layer_list = [], channel_list = [5, 10], linear_list = [100], dropout = 0.1):
        super(MyModel, self).__init__()
        self.layers = []
        count_conv = 0 #count of convolution layer
        count_linear = 0 #count of full connection layer
        input_size = 32 #input size of linear layer, begin as height and width of CIFAR-10 image

        channel_list = [3] + channel_list
        for layer in layer_list:
            #Activation Functions:
            if layer == 'ReLU':
                self.layers.append(nn.ReLU())
            elif layer == 'Iden': #Identity function f(x) = x
                pass
            elif layer == 'Logi':
                self.layers.append(nn.LogSigmoid())
            elif layer == 'Tanh':
                self.layers.append(nn.Tanh())
            #Pooling Methods
            elif layer == 'MaxPool':
                self.layers.append(nn.MaxPool2d(kernel_size=2))
                input_size //= 2
            elif layer == 'AvgPool':
                self.layers.append(nn.AvgPool2d(kernel_size=2))
                input_size //= 2
            #Normalization
            elif layer == 'Norm':
                if count_linear == 0:
                    self.layers.append(nn.BatchNorm2d(channel_list[count_conv]))
                else:
                    self.layers.append(nn.BatchNorm1d(linear_list[count_linear]))
            #Dropout
            elif layer == 'Drop':
                print(f"Dropout = {dropout}")
                self.layers.append(nn.Dropout(dropout))
            #Convolution layer
            elif layer == 'Conv':
                self.layers.append(nn.Conv2d(channel_list[count_conv], channel_list[count_conv + 1], kernel_size=kernel_size))
                input_size -= (kernel_size - 1)
                count_conv += 1
            #Flatten
            elif layer == 'Flatten':
                self.layers.append(nn.Flatten())
                input_size = input_size**2
                input_size *= channel_list[-1]
                linear_list = [input_size] + linear_list + [10]
                print(f"Flatten to {input_size} features")
            #Full connection
            elif layer == 'Linear':
                self.layers.append(nn.Linear(linear_list[count_linear], linear_list[count_linear + 1]))
                count_linear += 1
            elif layer == 'Transformer':
                print(f"Transformer with d_model = {linear_list[count_linear]}")
                encoder_layer = nn.TransformerEncoderLayer(d_model=linear_list[count_linear], nhead=8, batch_first=True)
                self.layers.append(nn.TransformerEncoder(encoder_layer, num_layers=6))
            else:
                raise "No such layer!"
        self.layers = nn.ModuleList(self.layers)

    def forward(self, input):
        
        for layer in self.layers:
            input = layer(input)

        return input
