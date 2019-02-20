import imageio
import pandas as pd
import numpy as np
import sys

def image_to_array(filename):
    # function that takes in the filename of an image, and returns the image as a flattened numpy array
    img = imageio.imread(filename)
    return np.array(img.flatten())


def images_to_XY(df, dir):
    # function that takes in a dataframe of labels and a directory of images and returns two matrices,
    # X is the pixels transformed to numerical values, Y is the label of that training example
    m = len(df)
    n = len(image_to_array(dir + df['id'].iloc[0] + '.tif'))
    X = np.zeros((m,n))
    Y = np.zeros((m,1))
    for i, name in enumerate(df.id):
        X[i,:] = image_to_array(dir + name + '.tif')
        Y[i] = df.label.iloc[i]

    return X, Y


def images_to_csv(df, dir, filename):
    # function that stores the image data in a csv file

    new_df = df['id'].copy()

    X,_ = images_to_XY(df, dir)
    new_df = pd.concat([new_df, pd.DataFrame(X)], axis=1)

    new_df.to_csv(filename + '.csv', index=False)

    return None


def load_XY(filename_X, filename_Y):
    # function that loads a csv containing pixel values, and a csv mapping an image to a label
    # and returns two matrices X and Y
    X_df = pd.read_csv(filename_X)
    Y_df = pd.read_csv(filename_Y)

    assert(X_df.id.equals(Y_df.id))

    return np.array(X_df.drop('id', 1)), np.array(Y_df.drop('id', 1))


if __name__ == '__main__':
    assert(len(sys.argv) > 1)
    file = sys.argv[1]
    x = image_to_array(file)
    print(x.shape)