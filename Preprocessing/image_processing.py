import numpy as np

def crop_center(array, dim, n):
    # function that takes in a flattened array, the original dimensions and returns
    # a flattened array that contains the center of the image that is n by n pixels

    assert(dim[0] >= n and dim[1] >= n)

    reshaped_array = array.reshape(dim)

    # find the upper left corner
    x_up = dim[0] // 2 - n // 2
    y_up = dim[1] // 2 - n // 2
    # find lower right corner
    x_down = dim[0] // 2 + n // 2
    y_down = dim[1] // 2 + n // 2

    return reshaped_array[x_up:x_down, y_up:y_down, :].flatten()


def grayscale_flat(array):
    # function that grayscales a flattened array

    gs = np.zeros((array.shape[0] // 3, 1))
    for i, j in enumerate(range(0, array.shape[0], 3)):
        gs[i] = int(np.mean(array[j:j+3]))

    return gs


def grayscale_matrix(array):
    # function that grayscales an array that is n x m x 3

    gs = int(np.mean(array, axis=2))

    return gs