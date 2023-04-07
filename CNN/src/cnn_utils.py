import numpy as np

import matplotlib.pyplot as plt


# Turns a dictionary into a class
class Dict2Class():
    
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])
            

@staticmethod
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)





def plot_training_results(results):
    
    x = list(range(1, len(results)+1))
    
    losses = [ tup[0] for tup in results ]
    acc_train = [ tup[1] for tup in results ]
    acc_test = [ tup[2] for tup in results ]

    # Convert losses to numpy array
    losses = np.asarray(losses)
    # Normalize losses so they match the scale in the plot (we are only interested in the trend of the losses!)
    losses = losses/np.max(losses)

    plt.figure()

    plt.plot(x, losses, lw=3)
    plt.plot(x, acc_train, lw=3)
    plt.plot(x, acc_test, lw=3)

    font_axes = {'family':'serif','color':'black','size':16}

    plt.gca().set_xticks(x)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel("Epoch", fontdict=font_axes)
    plt.ylabel("F1 Score", fontdict=font_axes)
    plt.legend(['Loss', 'F1 (train)', 'F1 (test)'], loc='lower left', fontsize=16)
    plt.tight_layout()
    plt.show()

