from extract_features import extract_features as ef
from random import shuffle
from numpy  import zeros

class validation_mc:
    """
This class is used to implement the K-fold validation process.
It uses the `fetch_train()` method in the `extract_features` class,
and then breaks the data into two separate parts, training and validation.

It implements the K-fold validation with a default K set to 11. This means
that it returns 11 sets of training and validation data.
"""
    @staticmethod
    def internal_fetch():
        """
An internal method used to fetch data using the `extract_features` module.
It is not required, since it is called anyway in the `fetch_data` method.
"""
        X_tot, y_tot = ef.fetch_train()
        X_sections = []
        y_sections = []
        for i in range(11):
            X_sections.append( X_tot[i*8 : i*8 + 8] )
            y_sections.append( y_tot[i*8 : i*8 + 8] )
        return (X_sections, y_sections, X_tot.shape[1])

    @staticmethod
    def fetch_data(val_num = 0):
        """
Returns training an validation data in the form of:
( (X_train, y_train) , (X_val, y_val) )
"""
        X_sections, y_sections, num_params = validation.internal_fetch()
        X_train = []
        y_train = []
        X_val   = []
        y_val   = []
        for i in range(11):
            if i == val_num:
                X_val = X_sections[i]
                y_val = y_sections[i]
            else:
                X_train.append( X_sections[i] )
                y_train += list( y_sections[i] )
        Xtr = zeros([80, num_params])
        for i in range(10):
            for j in range(8):
                for k in range(num_params):
                    Xtr[i * 8 + j,k] = X_train[i][j,k]
        del X_train
        X_train = Xtr
        y_val = list( y_val )
        return ( (X_train, y_train) , (X_val, y_val) )
