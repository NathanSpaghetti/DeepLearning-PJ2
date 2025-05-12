import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader

import model as m
import pickle

def train(train_set : DataLoader, valid_set,kernel_size = 5, 
          channel_list = [5, 10],pool = 'Avg', linear_list = [100],act = 'ReLU',
          optim = 'SGD', loss_func = 'CrossEntropy', num_epochs = 10)-> nn.Module:
    """
    :param valid_set: a tuple (tensor, tensor) of data and label for valid set
    """
    if torch.cuda.is_available():
        print("check device...CUDA available.")
        device = torch.device('cuda:0')
    else:
        print("using cpu as device.")
        device = torch.device('cpu')

    model = m.MyModel(kernel_size, channel_list, pool, linear_list, act)
    if optim == 'SGD':
        #optimizer = torch.optim.SGD(model.parameters(), lr = 0.06, weight_decay=.001)
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.06)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    if loss_func == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    model.train()
    loss_list = []
    for t in range(num_epochs):
        total_loss = 0
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
                #print(f"output is:{pred}")
                #valid
                valid = valid_set[0].to(device)
                label = valid_set[1].to(device)
                pred = model(valid)
                loss_valid = criterion(pred, label)
                print(f"Epoch [{t + 1}], iteration {count + 1}, loss = {loss.item()}, valid loss = {loss_valid.item()}")

        print(f"Epoch [{t + 1}], loss = {total_loss / len(train_set)}")
        #print(f"output is:{pred}")
        loss_list.append(total_loss / len(train_set))

    return model, loss_list

if __name__ == '__main__':
    print("Loading dataset...")
    train_set, test_set = m.get_train_test_set(batch_size=32)
    _, valid_set = enumerate(test_set).__next__()
    #model, loss = train(train_set, optim='SGD',pool='Max',channel_list=[18, 48], linear_list=[800], num_epochs=20)
    model, loss = train(train_set, valid_set, optim='SGD',pool='Max',channel_list=[15, 30], linear_list=[800], num_epochs=10)
    #save model if needed
    with open('./models/model.pickle', 'wb') as fp:
        pickle.dump(model, fp)
    with open('./figures/loss.pickle', 'wb') as fp:
        pickle.dump(loss, fp)
