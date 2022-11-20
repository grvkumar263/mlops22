from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

def test_is_split_same():
    # TEST if randome state is set
    # fractions
    train_frac = 0.8
    test_frac = 0.1
    dev_frac = 0.1
    
    # actual data
    d = datasets.load_digits()
    label = d.target
    data = d.images.reshape((len(d.images), -1))
    
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        data, label, test_size=test_frac, shuffle=True, random_state = 10
    )
    
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        data, label, test_size=test_frac, shuffle=True, random_state = 10
    )

    X_train3, X_test3, y_train3, y_test3 = train_test_split(
        data, label, test_size=test_frac, shuffle=True, random_state = 7
    )
    
    # checking
    assert (X_train1 == X_train2).all()
    assert (X_test1 == X_test2).all()
    assert (y_train1 == y_train2).all()
    assert (y_test1 == y_test2).all()
 
    assert (X_train3 != X_train2).any()
    assert (X_test3 != X_test2).any()
    assert (y_train3 != y_train2).any()
    assert (y_test3 != y_test2).any()
