import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

import model as m
import pickle

def test_model(model : nn.Module, test_set : DataLoader):
    model.eval()
    if torch.cuda.is_available():
        print("check device...CUDA available.")
        device = torch.device('cuda:0')
    else:
        print("using cpu as device.")
        device = torch.device('cpu')

    model = model.to(device)

    with torch.no_grad():
        correct_count = 0
        total = 0
        for count, (input, label) in enumerate(test_set):
            input = input.to(device)
            label = label.to(device)
            pred = model(input)
            pred = np.argmax(pred.cpu().detach().numpy(), axis = 1)
            acc = sum([pred[i] == label[i] for i in range(len(pred))])

            correct_count += acc
            total += len(pred)
    
    return correct_count / total

if __name__ == '__main__':
    with open('./models/model.pickle', 'rb') as fp:
        model = pickle.load(fp)
        train_set, test_set = m.get_train_test_set(batch_size=64)
        
        accuracy = test_model(model, test_set)
        print(f"Accuracy on test set:{accuracy}")
