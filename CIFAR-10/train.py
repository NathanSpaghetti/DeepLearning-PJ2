import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader

import model as m
import pickle

def train(train_set : DataLoader,
          valid_set,
          model = None,
          all_loss = None,
          lr = None,
          kernel_size = 3, 
          channel_list = [5, 10],
          layers = None,
          #pool = 'Avg',
          linear_list = [100],
          #act = 'ReLU',
          optim = 'SGD',
          loss_func = 'CrossEntropy',
          weight_decay = 0,
          dropout = 0.1,
          num_epochs = 10)-> nn.Module:
    """
    :param mode: a pre trained model, set to None for training from the begining
    :param valid_set: a tuple (tensor, tensor) of data and label for valid set
    :return loss_list: a list of loss at each iteration
    :return loss_list_epoch: a list of loss at each epoch
    :return loss_list_valid: a list of loss in valid set at each epoch
    """
    if torch.cuda.is_available():
        print("check device...CUDA available.")
        device = torch.device('cuda:0')
    else:
        print("using cpu as device.")
        device = torch.device('cpu')

    if model is None:
        model = m.MyModel(kernel_size, layers, channel_list, linear_list, dropout=dropout)

    if optim == 'SGD':
        print(f"Use SGD with lr = {0.06}")
        if lr is None:
            lr = 0.06
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay=weight_decay)
    elif optim == 'Adam':
        if lr is None:
            lr = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
    elif optim == 'Adadelta':
        if lr is None:
            lr = 1
        optimizer = torch.optim.Adadelta(model.parameters(), lr = lr, weight_decay=weight_decay)
    elif optim == 'Momentum':
        if lr is None:
            lr = 0.01
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay=weight_decay, momentum=0.9)

    if loss_func == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    model.train()
    if all_loss is None:
        loss_list = []
        loss_list_epoch = []
        loss_list_valid = []
    else:
        loss_list = all_loss[0]
        loss_list_epoch = all_loss[1]
        loss_list_valid = all_loss[2]

    for t in range(num_epochs):
        total_loss = 0
        total_loss_valid = 0
        for count, (input, label) in enumerate(train_set):
            optimizer.zero_grad()
            input = input.to(device)
            label = label.to(device)
            pred = model(input)

            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
           

            if (count + 1) % 100 == 0:
                model.eval()
                valid = valid_set[0].to(device)
                label = valid_set[1].to(device)
                pred = model(valid)
                loss_valid = criterion(pred, label)
                total_loss_valid += loss_valid.item()
                print(f"Epoch [{t + 1}], iteration {count + 1}, loss = {loss.item()}, valid loss = {loss_valid.item()}")
                loss_list.append(loss.item())
                model.train()

        print(f"Epoch [{t + 1}], loss = {total_loss / len(train_set)}, valid loss = {total_loss_valid / 15}")
        loss_list_epoch.append(total_loss / len(train_set))
        loss_list_valid.append(total_loss_valid / 15)

    return model, loss_list, loss_list_epoch, loss_list_valid

if __name__ == '__main__':
    print("Loading dataset...")
    train_set, test_set = m.get_train_test_set(batch_size=32)
    _, valid_set = enumerate(test_set).__next__()
    model, loss, loss_epoch, loss_valid = train(train_set,
                                                valid_set,
                                                optim='SGD',
                                                layers=["Conv", "ReLU","MaxPool", "Norm",
                                                        #"Conv", "ReLU","MaxPool", "Norm",
                                                        "Conv", "ReLU","Drop","Norm",
                                                        "Conv", "ReLU","Drop","Norm",
                                                        "Conv", "ReLU","MaxPool","Norm",
                                                        "Flatten",
                                                        "Linear","Norm", "ReLU","Drop",
                                                        "Linear","Norm", "ReLU","Drop",
                                                        "Linear","Norm", "ReLU","Drop",
                                                        "Linear"
                                                        ],
                                                #channel_list=[64, 64, 128, 256],#[16, 64, 128],
                                                channel_list=[64, 256,256, 512],
                                               
                                                linear_list=[4096, 2048, 1024],
                                                
                                                num_epochs=30,
                                                dropout=0.3,
                                                lr = 0.06,
                                                model=None,
                                                all_loss=None)
    #save model if needed
    with open('./models/model.pickle', 'wb') as fp:
        pickle.dump(model, fp)
    with open('./figures/loss.pickle', 'wb') as fp:
        pickle.dump((loss, loss_epoch, loss_valid), fp)
