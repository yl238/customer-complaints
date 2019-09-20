import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2, font='Arial')


def plot_confusion_matrix(cf, scale_by='Recall', figname='../figure/cf.png'):
    """
    The n-th attempt to create a function to draw a confusion matrix.
    Note that there is a bug in matplotlib 3.1.1 such that the top and bottom
    of the Seaborn heatmap are chopped off. Before it's fixed in 3.1.2 
    just revert back to an older version of Matplotlib.
    """
    if scale_by == 'Recall':
        recall_scale = np.sum(cf, axis=1)
        scaled = cf / recall_scale
    elif scale_by == 'Precision':
        precision_scale = np.sum(cf, axis=0)
        scaled = cf / precision_scale
    else:
        scaled = cf
        
    f, ax = plt.subplots(figsize=(8, 8))
    ax = sns.heatmap(scaled, cmap='Blues', cbar=None, annot=True, fmt=".2f")
    ax.set_xticklabels(shortened, rotation=60)
    ax.set_yticklabels(shortened, rotation=0);
    f.savefig(figname, bbox_inches='tight', dpi=300)