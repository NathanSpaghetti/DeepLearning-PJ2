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




if __name__ == '__main__':
    fig, axe = plt.subplots()
    single_plot("./figures/loss.pickle", axe)
    plt.show()