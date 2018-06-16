# https://blog.csdn.net/theonegis/article/details/51037850

''' This file is used to generate animation'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig = plt.figure()
data = np.random.random((255, 255))
im = plt.imshow(data, cmap='gray')

def animate(i):
    data = np.random.random((255, 255))
    im.set_array(data)
    return [im]

def main():
    anim = animation.FuncAnimation(fig, animate, frames=200, interval=60, blit=True)
    plt.show()

if __name__ == '__main__':
    main()