import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
import pickle
#from tqdm import tqdm as tqdm
#from IPython import display


from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader

if __name__ == '__main__':
    # ## Constants (parameters) initialization
    print("begin.")
    print(__name__)
    device_id = [0,1,2,3]
    num_workers = 4
    batch_size = 128

    # add our package dir to path 
    module_path = os.path.dirname(os.getcwd())
    home_path = module_path
    figures_path = os.path.join(home_path,'VGG_BatchNorm', 'reports', 'figures')
    models_path = os.path.join(home_path, 'VGG_BatchNorm', 'reports', 'models')
    text_path   = os.path.join(home_path, 'VGG_BatchNorm', 'reports', 'text')

    # Make sure you are using the right device.
    device_id = device_id
    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("check device...CUDA available.")
        device = torch.device('cuda:0')
    else:
        print("using cpu as device.")
        device = torch.device('cpu')
    print(device)
    print(torch.cuda.get_device_name(0))



    # Initialize your data loader and
    # make sure that dataloader works
    # as expected by observing one
    # sample from it.
    print("Loading train set")
    train_loader = get_cifar_loader(root='../data/',train=True)
    print("Loading test set")
    val_loader = get_cifar_loader(root='../data/',train=False)

    #for X,y in train_loader:
        ## --------------------
        # Add code as needed
        #
        #
        #
        #
        ## --------------------
    #    break



# This function is used to calculate the accuracy of model classification
def get_accuracy(model : nn.Module, test_set, device, total_count = -1):
    """
    total_count*batch_size of data points will be used to calculate the accuracy,
    and the accuracy of the whole set will be calculated if total_count == -1
    """
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        correct_count = 0
        total = 0
        for count, (input, label) in enumerate(test_set):
            input = input.to(device)
            label = label.to(device)
            pred = model(input)
            pred = torch.argmax(pred, dim = 1)
            #pred = np.argmax(pred.cpu().detach().numpy(), axis = 1)
            acc = sum([pred[i] == label[i] for i in range(len(pred))])

            correct_count += acc
            total += len(pred)

            if total_count > 0 and count == total_count:
                break
    
    return correct_count / total

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    print("Train begin...")
    #for epoch in tqdm(range(epochs_n), unit='epoch'):
    for epoch in range(epochs_n):
        if scheduler is not None:
            scheduler.step()
        model.train()
        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        count = 0
        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            ## --------------------
            # Add your code
            loss_list.append(loss.item())
            learning_curve[epoch] += loss.item()
            #
            ## --------------------

            loss.backward()
            optimizer.step()
            grad.append(model.classifier[4].weight.grad.clone().cpu())
            count += 1
            if (count + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}], iteration [{count + 1}]: loss = {loss.item()}")

        losses_list.append(loss_list)
        grads.append(grad)
        #display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))

        learning_curve[epoch] /= batches_n
        axes[0].plot(learning_curve)

        # Test your model and save figure here (not required)
        # remember to use model.eval()
        ## --------------------
        # Add code as needed
        val_accuracy_curve[epoch] = get_accuracy(model, val_loader,  device, 4)
        train_accuracy_curve[epoch]=get_accuracy(model, train_loader,device, 4)
        print(f"Epoch:{epoch + 1}, loss = {learning_curve[epoch]}, train accuracy = {train_accuracy_curve[epoch]}, validation accuracy = {val_accuracy_curve[epoch]}")
        if max_val_accuracy < val_accuracy_curve[epoch]:
            print(f"New best accuracy at epoch {epoch + 1}")
            max_val_accuracy = val_accuracy_curve[epoch]
            max_val_accuracy_epoch = epoch
        model.train()
        #
        ## --------------------
    
    with open(os.path.join(models_path, 'model_001.pickle'), 'wb') as f:
        pickle.dump(model, f)
    return model, losses_list, grads, train_accuracy_curve, val_accuracy_curve

if __name__ == '__main__':
    # Train your model
    # feel free to modify
    epo = 20

    set_random_seeds(seed_value=2020, device=device)
    model = VGG_A()
    #model = VGG_A_BatchNorm()
    lr = 0.001
    lr = 0.01
    #optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()

    model, loss, grads, tain_accu, val_accu = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
    np.savetxt(os.path.join(text_path, 'loss_001.txt'), loss, fmt='%s', delimiter=' ')
    #np.savetxt(os.path.join(text_path, 'grads.txt'), grads, fmt='%s', delimiter=' ')

    with open(os.path.join(figures_path, 'loss_001.pickle'), 'wb') as f:
        pickle.dump(loss, f)

    """with open(os.path.join(figures_path, 'train_accu_3e-3.pickle'), 'wb') as f:
        pickle.dump(tain_accu, f)

    with open(os.path.join(figures_path, 'valid_accu_3e-3.pickle'), 'wb') as f:
        pickle.dump(val_accu, f)"""

    with open(os.path.join(figures_path, 'grad_001.pickle'), 'wb') as f:
        pickle.dump(grads, f)
    
    """with open(os.path.join(models_path, 'model_norm.pickle'),'rb') as f:
        model = pickle.load(f)
    
    ac1 = get_accuracy(model, train_loader, device)
    ac2 = get_accuracy(model, val_loader, device)
    print(f"Train accuracy:{ac1}, val accuracy:{ac2}")"""
    # Maintain two lists: max_curve and min_curve,
    # select the maximum value of loss in all models
    # on the same step, add it to max_curve, and
    # the minimum value to min_curve
    min_curve = []
    max_curve = []
    ## --------------------
    # Add your code
    
def get_minxmax_curve( root = './reports/figures/'):
    #prepaing data: VGG_A
    with open(os.path.join(root, 'loss.pickle'), 'rb') as f:
        loss1 = pickle.load(f)
    with open(os.path.join(root, 'loss_2e-3.pickle'), 'rb') as f:
        loss2 = pickle.load(f)
    with open(os.path.join(root, 'loss_1e-4.pickle'), 'rb') as f:
        loss3 = pickle.load(f)
    with open(os.path.join(root, 'loss_5e-4.pickle'), 'rb') as f:
        loss4 = pickle.load(f)
        
    loss_all = [loss1, loss2, loss3, loss4]
    loss     = [[],[],[],[]]

    with open(os.path.join(root, 'loss_norm.pickle'), 'rb') as f:
        loss1 = pickle.load(f)
    with open(os.path.join(root, 'loss_norm_2e-3.pickle'), 'rb') as f:
        loss2 = pickle.load(f)
    with open(os.path.join(root, 'loss_norm_1e-4.pickle'), 'rb') as f:
        loss3 = pickle.load(f)
    with open(os.path.join(root, 'loss_norm_5e-4.pickle'), 'rb') as f:
        loss4 = pickle.load(f)

    loss_norm_all = [loss1, loss2, loss3, loss4]
    loss_norm     = [[],[],[],[]]

    for i in range(4):
        for l in loss_all[i]:
            loss[i] += l
        for l in loss_norm_all[i]:
            loss_norm[i] += l

    min_curve = []
    max_curve = []
    min_curve_norm = []
    max_curve_norm = []

    for i in range(len(loss[0])):
        min_curve.append(min(loss[0][i], loss[1][i], loss[2][i], loss[3][i]))
        max_curve.append(max(loss[0][i], loss[1][i], loss[2][i], loss[3][i]))

        min_curve_norm.append(min(loss_norm[0][i], loss_norm[1][i], loss_norm[2][i], loss_norm[3][i]))
        max_curve_norm.append(max(loss_norm[0][i], loss_norm[1][i], loss_norm[2][i], loss_norm[3][i]))

    return min_curve, max_curve, min_curve_norm, max_curve_norm

    #
    #
    ## --------------------

    # Use this function to plot the final loss landscape,
    # fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape(axe, stepsize = 1):
        ## --------------------
        # Add your code

    min_curve, max_curve, min_curve_norm, max_curve_norm = get_minxmax_curve()
    iters = [i for i in range(0, len(max_curve), stepsize)]
    axe.set_ylabel("loss")
    axe.set_xlabel("iteration")
    axe.fill_between(iters, max_curve[::stepsize], min_curve[::stepsize], color = 'steelblue', label = 'VGG', alpha = 0.5)
    axe.fill_between(iters, min_curve_norm[::stepsize], max_curve_norm[::stepsize], color = 'lightgreen', label = 'VGG with BatchNorm', alpha = 0.5)
    axe.legend()

        #
        #
        ## --------------------