

import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def kde_stack(x1, y1, x2, y2, h = 6, w = 10, title=None, label=None):
    """
    Function to plot joint kde and marginals together
    """
    
    cmaps = ['Reds', 'Blues']
    g = sns.JointGrid(x=x1, y=y1)
    plt.subplots_adjust(top=0.9)
    if title != None:
        g.fig.suptitle(title)
    g.fig.set_figwidth(w)
    g.fig.set_figheight(h)
    sns.kdeplot(x1, y1, cmap=cmaps[0], ax=g.ax_joint)
    sns.kdeplot(x2, y2, cmap=cmaps[1], ax=g.ax_joint)
    
    if label != None:
        lbs = label
        label_patches = []
        label_patches.append(mpatches.Patch(color=sns.color_palette(cmaps[0])[2], label=lbs[0]))
        label_patches.append(mpatches.Patch(color=sns.color_palette(cmaps[1])[2], label=lbs[1]))
        plt.legend(handles=label_patches, loc='upper left');
    sns.distplot(x1, color="r", ax=g.ax_marg_x, hist=False, kde=True)
    sns.distplot(x2, color="b", ax=g.ax_marg_x, hist=False, kde=True)
    sns.distplot(y1, color="r", ax=g.ax_marg_y, hist=False, kde=True, vertical=True)
    sns.distplot(y2, color="b", ax=g.ax_marg_y, hist=False, kde=True, vertical=True)
    pass
    
