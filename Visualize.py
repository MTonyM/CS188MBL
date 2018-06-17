# https://blog.csdn.net/theonegis/article/details/51037850

''' This file is used to generate animation'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

SIZE_MIN = 50
SIZE_MAX = 50 ** 2
NS = 50

pos = np.random.uniform(0, 1, (NS, 2))
color = np.ones((NS, 4)) * (0, 0, 0, 1)

fig = plt.figure(num = "Bike Simulation", figsize=(6,6), facecolor='white')

def init():
    global size, scat

    # New axis over the whole figure, no frame and a 1:1 aspect ratio
    ax = fig.add_axes([0, 0, 1, 1], frameon = False, aspect = 1)

    # Ring sizes
    size = np.linspace(SIZE_MIN, SIZE_MAX, NS)

    # Scatter plot
    scat = ax.scatter(pos[:,0], pos[:,1], s=size, lw=0.5, edgecolors=color, facecolors=color)

    # Ensure limits are [0,1] and remove ticks
    ax.set_xlim(0, 1), ax.set_xticks([])
    ax.set_ylim(0, 1), ax.set_yticks([])

def update(frame):
    global size

    # Each ring is made larger
    size += (SIZE_MAX - SIZE_MIN) / NS

    # Reset specific ring
    i = frame % 50
    size[i] = SIZE_MIN

    # Update scatter object
    scat.set_edgecolors(color)
    scat.set_sizes(size)
    scat.set_offsets(pos)

    # Return the modified object
    return scat,

def showAnimation():
    init()
    anim = animation.FuncAnimation(fig, update, interval = 10, blit = True, frames = 200)
    plt.show()

def main():
    showAnimation()

if __name__ == '__main__':
    main()