import pickle
import matplotlib.pyplot as plt

def single_plot(filename, axe):
    """
    Plot the loss curve of a single model
    """

    with open(filename, 'rb') as fp:
        (loss, loss_epoch, loss_valid) = pickle.load(fp)

    iterations = [i for i in range(len(loss))]
    epochs     = [i for i in range(len(loss_epoch))]
    
    axe.set_ylabel("loss")
    axe.set_xlabel("iteration")
    axe.set_title("")
    p1, = axe.plot(iterations, loss, label = "Train loss")

    axe1 = axe.twiny()
    axe1.set_xlabel("Epoch")
    p2, = axe1.plot(epochs, loss_epoch,color = 'green', label = "Train loss at each epoch")
    p3, = axe1.plot(epochs, loss_valid,color = 'red', label = "Valid loss at each epoch")

    #draw legend
    figs = [p1, p2, p3]
    axe.legend(figs, [i.get_label() for i in figs])


def multi_plot(name_list, axe, folder = './figures/', labels = []):
    """
    :param name_list: a list of every file storing loss information need to be shown
    :param labels: a list storing description of each model
    """
    loss_epochs = []
    valid_epochs = []
    figures_loss = []
    figures_valid = []

    palette1 = ['forestgreen','lightseagreen','steelblue','purple']
    palette2 = ['darkred','darkorange','gold','olive']
    count = 0

    for name in name_list:
        with open(folder + name, 'rb') as fp:
            (_, loss_epoch, valid_epoch) = pickle.load(fp)
            loss_epochs.append(loss_epoch)
            valid_epochs.append(valid_epoch)
    
    axe.set_ylabel("loss")
    axe.set_xlabel("iteration")
    axe.set_title("")
    for i in range(len(name_list)):
        loss_epoch = loss_epochs[i]
        iterations = [i for i in range(len(loss_epoch))]
        p, = axe.plot(iterations, loss_epoch,color = palette1[i], label = "Loss of " + labels[i])
        figures_loss.append(p)

    axe1 = axe.twiny()
    for i in range(len(name_list)):
        valid_epoch = valid_epochs[i]
        iterations = [i for i in range(len(valid_epoch))]
        p, = axe1.plot(iterations, valid_epoch, color = palette2[i], label = "Valid loss of " + labels[i], linestyle='dashed')
        figures_valid.append(p)
        
    axe.legend(figures_loss, [i.get_label() for i in figures_loss])
    axe1.legend(figures_valid, [i.get_label() for i in figures_valid],loc = 'upper left')




if __name__ == '__main__':
    fig, axe = plt.subplots()
    single_plot("./figures/final120.pickle", axe)
    #multi_plot(["base.pickle", "conv2.pickle", "conv4.pickle" ], axe, labels=["base","conv2","conv4"])
    #multi_plot(["base.pickle", "convbig.pickle", "convsmall.pickle" ], axe, labels=["base","convbig","convsmall"])
    #multi_plot(["base.pickle", "500.pickle", "4000.pickle" ], axe, labels=["base","500","4000"])
    #multi_plot(["base.pickle", "500.pickle", "4000.pickle" ], axe, labels=["base","500","4000"])
    #multi_plot(["basen.pickle", "wd01.pickle", "wd03.pickle", "wd05.pickle"], axe, labels=["base","wd=1e-3","wd=3e-3", "wd=5e-3"])
    #multi_plot(["basen.pickle", "dp3.pickle", "dp5.pickle" ], axe, labels=["base","dropout = 0.3","dropout = 0.5"])
    #multi_plot(["basen.pickle", "iden.pickle", "logi.pickle", "tanh.pickle" ], axe, labels=["ReLU","Identity","Log-sigmoid","tanh"])
    #multi_plot(["basen.pickle", "iden.pickle", "logi.pickle", "mix2.pickle" ], axe, labels=["ReLU","Identity","Log-sigmoid","Mix"])
    #multi_plot(["basen.pickle", "momentum.pickle", "adam.pickle" ], axe, labels=["SGD","Momentum", "Adam"])
    #multi_plot(["final.pickle", "final120.pickle" ], axe, labels=["final","final2"])
    #multi_plot(["basen.pickle", "trans.pickle"], axe, labels=["base","trans"])

    plt.show()