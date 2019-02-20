import Preprocessing.preprocessing as pp
import numpy as np
from sklearn.linear_model import LogisticRegression

train_csv_x = '/Users/joelpersson/Documents/GitHub/CS229A/Data/train_images.csv'
train_csv_y = '/Users/joelpersson/Documents/GitHub//CS229A/Data/train_labels.csv'

X, Y = pp.load_XY(train_csv_x, train_csv_y)

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, Y)