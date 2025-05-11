import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader

import model as m

def train(train_set : DataLoader,kernel_size = 5, 
          channel_list = [5, 10],pool = 'Avg', linear_list = [100],act = 'ReLU',
          optim = 'SGD', loss_func = 'CrossEntropy', num_epochs = 10)-> nn.Module:
    
    if torch.cuda.is_available():
        print("check device...CUDA available.")
        device = torch.device('cuda:0')
    else:
        print("using cpu as device.")
        device = torch.device('cpu')

    model = m.MyModel(kernel_size, channel_list, pool, linear_list, act)
    if optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.06)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    if loss_func == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    model.train()
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
                print(f"Epoch [{t + 1}], iteration {count + 1}, loss = {loss.item()}")

        print(f"Epoch [{t + 1}], loss = {total_loss / len(train_set)}")

    return model

if __name__ == '__main__':
    print("Loading dataset...")
    train_set, test_set = m.get_train_test_set(batch_size=64)
    model = train(train_set, optim='Adam',channel_list=[10, 30], linear_list=[1000])
