import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader

import model as m
import pickle



def save_chunk(filename, chunk_count = 4, new_name = []):
    assert len(new_name) >= chunk_count
    #resave data in filename in multiple chunks
    with open(filename, 'rb') as f:
        f.seek(0, os.SEEK_END)
        length = f.tell()
        chunk_size = length // chunk_count
        f.seek(0, os.SEEK_SET)

        for i in range(chunk_count - 1):
            with open(new_name[i], 'wb') as f2:
                data = f.read(chunk_size)
                f2.write(data)
        with open(new_name[chunk_count - 1], 'wb') as f2:
            data = f.read()
            f2.write(data)

def load_chunk(filenames = [], keepfile = False, temp_filename = ".tempfile"):
    """
    :param keepfile: keep the temp file
    :param temp_filename: filename of the temp file
    """
    result = b''
    for name in  filenames:
        with open(name, 'rb') as f:
            data = f.read()
            result += data
    
    with open(temp_filename, 'wb') as f:
        f.write(result)
    with open(temp_filename, 'rb') as f:
        result = pickle.load(f)
    if not keepfile:
        os.unlink(temp_filename)

    return result

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
    #save_chunk('./models/final.pickle', 4, ['./models/final1','./models/final2','./models/final3','./models/final4'])

    #load large models
    model = load_chunk(['./models/final1','./models/final2','./models/final3','./models/final4'],False ,'./models/.temp_file')
    train_set, test_set = m.get_train_test_set(batch_size=128)
        
    accuracy = test_model(model, test_set)
    print(f"Accuracy on test set:{accuracy}")

    with open('./models/base.pickle', 'rb') as fp:
        model = pickle.load(fp)
        train_set, test_set = m.get_train_test_set(batch_size=128)
        
        accuracy = test_model(model, train_set)
        print(f"Accuracy on test set:{accuracy}")

        
