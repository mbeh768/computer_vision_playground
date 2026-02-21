"""
Functions for visualization of outputs/processes
"""

import math
import matplotlib.pyplot as plt


def show_images(images, titles=None, cmaps=None, figsize_scale=4):
    """
    Display images in a near-square grid.

    Parameters
    ----------
    images : list of arrays/tensors
        Images to display.
    titles : list[str], optional
        Titles for each subplot.
    cmaps : list[str], optional
        Colormap per image.
    figsize_scale : int
        Size multiplier per subplot.
    """

    n = len(images)

    # choose grid close to square
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * figsize_scale,
                                      rows * figsize_scale))

    # axes handling for 1-row/1-col cases
    axes = axes.flatten() if n > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < n:
            img = images[i]

            cmap = cmaps[i] if cmaps else None
            title = titles[i] if titles else ""

            ax.imshow(img, cmap=cmap)
            ax.set_title(title)

    plt.tight_layout()
    plt.show()