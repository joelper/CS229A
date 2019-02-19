import numpy as np
import pandas as pd
from shutil import copy2


def sample_datasets(filename, n):
    # function that samples n datapoints from a csv
    df = pd.read_csv(filename)
    assert(len(df) >= n)

    return df.sample(n).reset_index(drop=True)


def split_data(df, frac_train, frac_val, frac_test):
    # function that splits a dataframe into train, validation and test set based of the fractions provided
    # assumes the dataframe is in a random order (i.e. no correlation between the ordering)
    assert(np.abs(frac_train + frac_val + frac_test - 1) < 1e-9)

    train_data = df[:int(frac_train*len(df))]
    val_data = df[len(train_data):int((frac_train + frac_val)*len(df))]
    test_data = df[int((frac_train + frac_val) * len(df)):]

    return train_data, val_data, test_data


def copy_images(df, curr_dir, new_dir):
    # function that copies the images in df from curr_dir to a new directory new_dir
    for name in df.id:
        copy2(curr_dir + name + '.tif', new_dir)


if __name__ == '__main__':
    filename = '/Users/joelpersson/Documents/GitHub/CS229A/Data/train_labels.csv'  # filename of csv with train labels
    n = 10000

    df = sample_datasets(filename, n)

    train_df, val_df, test_df = split_data(df, 0.6, 0.2, 0.2)
    copy_images(train_df, '/Users/joelpersson/Documents/GitHub/CS229A/Data/train/',
                '/Users/joelpersson/Documents/GitHub/CS229A/Data/train2')
    copy_images(val_df, '/Users/joelpersson/Documents/GitHub/CS229A/Data/train/',
                '/Users/joelpersson/Documents/GitHub/CS229A/Data/val')
    copy_images(test_df, '/Users/joelpersson/Documents/GitHub/CS229A/Data/train/',
                '/Users/joelpersson/Documents/GitHub/CS229A/Data/test')

    train_df.to_csv("train2_labels.csv")
    val_df.to_csv("val_labels.csv")
    test_df.to_csv("test_labels.csv")