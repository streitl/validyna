import numpy as np
import matplotlib.pyplot as plt
import os

def lcs(to_open='log.txt',
        things=['train_loss',
                'val_loss'],
        save_to='lcs.png'):
    curves = get_curves(to_open, things)
    plot_curves(curves, save_to)

def get_curves(filename, things):
    with open(filename, 'r') as f:
        train = []
        val = []
        for line in f:
            foo = line.split('\t')
            for bar in foo:
                if bar.startswith(things[0][:2]):
                    train.append(float(bar.split(': ')[1].strip()))
                elif bar.startswith(things[1][:2]):
                    val.append(float(bar.split(': ')[1].strip()))
    return train, val

def plot_curves(curves_dict, labels_dict={'x':'Epochs','y':'Loss'}, 
                filename=None, showplot=True):

    markers=["o",
             "v",
             "^",
             "<",
             ">",
             "8",
             "s",
             "*",
             "h",
             "H",
             "+",
             "x",
             "D",
             "d",
             "_",
             ".",
             ",",
             ]
    colours=["forestgreen", 
             "yellowgreen", #alternate ones are paired (light versions)
             "blueviolet"
             "mediumorchid",
             "orangered",
             "lightsalmon"
             "royalblue",
             "lightsteelblue"
             ]
    plt.figure()
    ax = plt.axes()
    ax.set_facecolor("gainsboro")
    # plt.title('Fooo', fontsize=22)
    plt.xlabel(labels_dict['x'])
    plt.ylabel(labels_dict['y'])
    
    # Lighten borders
    plt.gca().spines["top"].set_alpha(0.3)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0.3)
    plt.gca().spines["left"].set_alpha(.3)
    plt.grid(alpha=0.3)

    names = [name for name in curves_dict.keys()]
    curves = [curves_dict[name] for name in names]

    for i in range(len(curves)):
        plt.plot(curves[i], label=names[i], marker=markers[i], color=colours[i], linewidth=2)
        plt.legend()
    if filename==None or showplot:
        plt.show()
    else:
        plt.savefig(filename)
        if not showplot:
            plt.close()

def plot_sines(generated, groundtruth, filename):

    markers=["o",
             "v",
             "^",
             "<",
             ">",
             "8",
             "s",
             "*",
             "h",
             "H",
             "+",
             "x",
             "D",
             "d",
             "_",
             ".",
             ",",
             ]
    colours=["forestgreen", 
             #"yellowgreen", #alternate ones are paired (light versions)
             "blueviolet"
             "mediumorchid",
             "orangered",
             "lightsalmon"
             "royalblue",
             "lightsteelblue"
             ]
    plt.figure()
    ax = plt.axes()
    ax.set_facecolor("gainsboro")
    # plt.title('Fooo', fontsize=22)
    plt.xlabel(curves_dict['x'])
    plt.ylabel(curves_dict['y'])

    #plt.ylim(0, 90)    
    #plt.xlim(1968, 2014) 

    #plt.yticks(range(0, 91, 10), [str(x) + "%" for x in range(0, 91, 10)], fontsize=14)    
    #plt.xticks(fontsize=14)
    
    # Lighten borders
    plt.gca().spines["top"].set_alpha(0.3)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0.3)
    plt.gca().spines["left"].set_alpha(.3)
    plt.grid(alpha=0.3)

    plt.plot(groundtruth, label="Ground truth",color=colours[0], linewidth=2)
    plt.plot(generated, label="Generated",color=colours[1], linestyle=(0,(5,1)), linewidth=2)

    plt.legend()
    plt.savefig(filename)


def plot_curves_d(curves_dict, filename):

    markers=["o",
             "v",
             "^",
             "<",
             ">",
             "8",
             "s",
             "*",
             "h",
             "H",
             "+",
             "x",
             "D",
             "d",
             "_",
             ".",
             ",",
             ]
    colours=["forestgreen", 
             "yellowgreen", #alternate ones are paired (light versions)
             "blueviolet"
             "mediumorchid",
             "orangered",
             "lightsalmon"
             "royalblue",
             "lightsteelblue"
             ]
    plt.figure()
    ax = plt.axes()
    ax.set_facecolor("gainsboro")
    # plt.title('Fooo', fontsize=22)
    plt.xlabel(curves_dict['x'])
    plt.ylabel(curves_dict['y'])
    
    # Lighten borders
    plt.gca().spines["top"].set_alpha(0.3)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0.3)
    plt.gca().spines["left"].set_alpha(.3)
    plt.grid(alpha=0.3)

    for i in curves_dict:
        if i == "x" or i == "y" or i == "title":
            pass
        else:
            if i.lower().startswith('t'):
                plt.plot(curves_dict[i], label=i, color=colours[0], linewidth=2)
            else:
                plt.plot(curves_dict[i], label=i, color=colours[1], linewidth=2)

    plt.legend()
    plt.savefig(filename)

