from extract_features import extract_features as ef
from random import shuffle
from numpy  import zeros

class validation_jsrt:
    """
This class is used to implement the K-fold validation process.
It uses the `fetch_train()` method in the `extract_features` class,
and then breaks the data into two separate parts, training and validation.

It implements the K-fold validation with a default K set to 10. This means
that it returns 10 sets of training and validation data.
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
        for i in range(10):
            X_sections.append( X_tot[i*15 : i*15 + 15] )
            y_sections.append( y_tot[i*15 : i*15 + 15] )
        return (X_sections, y_sections, X_tot.shape[1])

    @staticmethod
    def fetch_data(val_num = 0):
        """
Returns training an validation data in the form of:
( (X_train, y_train) , (X_val, y_val) )
"""
        X_sections, y_sections, num_params = validation_jsrt.internal_fetch()
        X_train = []
        y_train = []
        X_val   = []
        y_val   = []
        for i in range(10):
            if i == val_num:
                X_val = X_sections[i]
                y_val = y_sections[i]
            else:
                X_train.append( X_sections[i] )
                y_train += list( y_sections[i] )
        Xtr = zeros([135, num_params])
        for i in range(9):
            for j in range(15):
                for k in range(num_params):
                    Xtr[i * 15 + j,k] = X_train[i][j,k]
        del X_train
        X_train = Xtr
        y_val = list( y_val )
        return ( (X_train, y_train) , (X_val, y_val) )

