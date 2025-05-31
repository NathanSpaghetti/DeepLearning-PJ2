import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import os


def plot_loss(axe, root = './reports/figures/'):
    with open(os.path.join(root, 'loss.pickle'), 'rb') as f:
        losses = pickle.load(f)
    with open(os.path.join(root, 'loss_norm.pickle'), 'rb') as f:
        losses_norm = pickle.load(f)
    
    axe.set_ylabel("loss"),
    axe.set_xlabel("iteration")
    axe.set_title("")

    loss_iter = []
    loss_iter_norm = []
    figures = []
    for i in range(len(losses)):
        loss_iter += losses[i]
        loss_iter_norm += losses_norm[i]

    #Plot loss at each iteration
    axe.set_ylabel("loss")
    axe.set_xlabel("iteration")
    iters = [i for i in range(len(loss_iter))]
    p, = axe.plot(iters, loss_iter, color = 'lightgreen', label = "Loss of VGG", alpha = 0.5)
    figures.append(p)
    p, = axe.plot(iters, loss_iter_norm, color = 'steelblue', label = "Loss of VGG with BatchNorm", alpha = 0.5)
    figures.append(p)

    axe.legend(figures, [i.get_label() for i in figures], loc = 'center right')

    #Plot loss at each epoch
    axe1 = axe.twiny()
    axe1.set_xlabel("epoch")
    loss_epoch = np.mean(losses, axis=1)
    loss_epoch_norm = np.mean(losses_norm, axis=1)
    iters = [i for i in range(len(loss_epoch))]
    figures = []
    p, = axe1.plot(iters, loss_epoch, color = 'green', label = "Epoch loss of VGG", lw = 0.5)
    figures.append(p)
    p, = axe1.plot(iters, loss_epoch_norm, color = 'blue', label = "Epoch loss of VGG with BatchNorm", lw = 0.5)
    figures.append(p)
    axe1.legend(figures, [i.get_label() for i in figures], loc = 'best')

def plot_accuracy(axe, root = './reports/figures/'):
    with open(os.path.join(root, 'train_accu.pickle'), 'rb') as f:
        train_accu = pickle.load(f)
        train_accu = torch.tensor(train_accu).cpu().view(-1)
    with open(os.path.join(root, 'valid_accu.pickle'), 'rb') as f:
        valid_accu = pickle.load(f)
        valid_accu = torch.tensor(valid_accu).cpu().view(-1)

    with open(os.path.join(root, 'train_accu_norm.pickle'), 'rb') as f:
        train_accu_norm = pickle.load(f)
        train_accu_norm = torch.tensor(train_accu_norm).cpu().view(-1)
    with open(os.path.join(root, 'valid_accu_norm.pickle'), 'rb') as f:
        valid_accu_norm = pickle.load(f)
        valid_accu_norm = torch.tensor(valid_accu_norm).cpu().view(-1)

    axe.set_ylabel("accuracy")
    axe.set_xlabel("epoch")
    iter = [i for i in range(len(train_accu_norm))]


    axe.plot(iter, train_accu,      color = 'lightgreen', label = 'Train accuracy of VGG')
    axe.plot(iter, train_accu_norm, color = 'steelblue' , label = 'Train accuracy of VGG with BatchNorm')
    
    axe.plot(iter, valid_accu,      color = 'green', label = 'Validation accuracy of VGG')
    axe.plot(iter, valid_accu_norm, color = 'blue' , label = 'Validation accuracy of VGG with BatchNorm')

    axe.legend()


def extract_grad(root = './reports/figures/'):
    """
    extract the grad into list from the storage file
    """
    with open(os.path.join(root, 'grad03.pickle'), 'rb') as f:
        grad1 = pickle.load(f)
    with open(os.path.join(root, 'grad_001.pickle'), 'rb') as f:
        grad2 = pickle.load(f)
        
    grad_all = [grad1, grad2]
    grad     = [[],[]]

    with open(os.path.join(root, 'grad_norm_03.pickle'), 'rb') as f:
        grad1 = pickle.load(f)
    with open(os.path.join(root, 'grad_norm_001.pickle'), 'rb') as f:
        grad2 = pickle.load(f)

    grad_all_norm = [grad1, grad2]
    grad_norm     = [[],[]]

    for l in grad_all[0]:
        grad[0] += l
    for l in grad_all[1]:
        grad[1] += l
    for l in grad_all_norm[0]:
        grad_norm[0] += l
    for l in grad_all_norm[1]:
        grad_norm[1] += l
    
    return grad, grad_norm

def plot_grad_change(axe, grad, grad_norm, stepsize = 1):
    """
    get the gradient predictiveness
    """
    
    
    grad_change_max = []
    grad_change_min = []
    grad_change_norm_max = []
    grad_change_norm_min = []

    for i in range(0, len(grad[0]) - 1):
        grad_change = [torch.norm(grad[0][i] -grad[0][i + 1], p=2), torch.norm(grad[1][i] -grad[1][i + 1], p=2), ]
        
        grad_change_norm = [torch.norm(grad_norm[0][i] -grad_norm[0][i + 1], p=2), torch.norm(grad_norm[1][i] -grad_norm[1][i + 1], p=2), ]
        
        grad_change_max.append( max(grad_change))
        grad_change_min.append( min(grad_change))
        grad_change_norm_max.append( max(grad_change_norm))
        grad_change_norm_min.append( min(grad_change_norm))
    
    iter = [i for i in range(0,len(grad_change_min), stepsize)]
    axe.set_ylabel("predictiveness")
    axe.set_xlabel("iteration")
    axe.fill_between(iter, grad_change_max[::stepsize],     grad_change_min[::stepsize],       
                     color = 'steelblue', label = 'VGG', alpha = 0.5)
    axe.fill_between(iter, grad_change_norm_max[::stepsize], grad_change_norm_min[::stepsize], 
                     color = 'lightgreen', label = 'VGG with BatchNorm', alpha = 0.5)
    axe.legend()

def plot_grad_diff(axe, grad, grad_norm, stepsize = 1):
    """
    get the maximum difference of gradient
    """
    grad_diff = []
    grad_diff_norm = []

    for i in range(len(grad[0])):
        grad_diff.append(torch.norm(grad[0][i] - grad[1][i], p = 2))
        grad_diff_norm.append(torch.norm(grad_norm[0][i] - grad_norm[1][i], p = 2))

    iter = [i for i in range(0,len(grad_diff), stepsize)]
    axe.set_ylabel("effectiveness")
    axe.set_xlabel("iteration")
    axe.plot(iter, grad_diff[::stepsize], color = 'steelblue', label = 'VGG')
    axe.plot(iter, grad_diff_norm[::stepsize], color = 'lightgreen', label = 'VGG with BatchNorm')
    axe.legend()


if __name__ == '__main__':
    fig, axe = plt.subplots()
    plt.ylim([0, 3])
    grad, grad_norm = extract_grad()
    plot_grad_change(axe, grad, grad_norm, 10)
    #plot_grad_diff(axe, grad, grad_norm, 10)
    plt.show()