#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:06:11 2018

@author: rajarshi
"""

import random  as rnd
import pandas  as pd
import numpy   as np

set_name = 'mc-cxr'

class extract_features:
    """
    This is a static class. There is no need to create objects of this class. The class has
    three static methods, which are:
        [1]  extract_features.prepare(case, option)
        [2]  extract_features.randomize(mat1, mat2)
        [3]  extract_features.separate(overall)
    """
    @staticmethod
    def prepare(case, option):
        """
        This function is used to extract feature vectors from the dataset pickles.
        Usage: `prepare(case, option)`
               where [1] case   -> 0 or 1, depending on the class
                     [2] option -> 'Train' or 'Test' depending on which set is required.
        """
        pfname = 'params' # Name of the parameter list file.
        
        prefix  = './'
        f = open(prefix + option + '_data/left-lung-' + str(case) + '.csv', 'r')
        table = pd.read_csv(f)
        f.close()
        
        f_table = []
        
        f = open(prefix + pfname + '.py', 'r')
        params = f.readlines()
        f.close()
        param_count = 0
        for param in params:
            if param[0] == '#':
                continue
            if '#' in param:
                param = param.split('#')[0]
            param_count += 1
            param = param[:len(param) - 1].strip()  # To get rid of the NewLine character at the end of the line.
            f_table.append( list( table[param] )  )
        
        f_table = np.matrix(f_table)
        f_table = f_table.transpose()
        #print 'Collecting ' + option + ' case for class ' + str(case) + ' using', param_count, 'parameters.'
        return f_table

    @staticmethod
    def mat_norm(mat):
        """
        Used for normalizing a matrix within the `fetch_train` method. I need a new one, since the old one
        works only on the list datatype, and not on np.matrix.
        """
        rows, cols = mat.shape
        nrm = np.zeros([rows,cols])
        for col in range(cols):
            target = np.array(list(mat[:,col]))
            t_mean = np.mean(target)
            t_vrnc = np.var(target)
            t_stdv = np.sqrt(t_vrnc)
            target = target - t_mean
            target = target / t_stdv
            for row in range(rows):
                nrm[row,col] = target[row]
        return nrm

    @staticmethod
    def fetch_train():
        """
        Reads the file Train_data/left-lung-shuffled.csv and extracts it, separating
        the data from the labels. It takes no arguments, and uses the exact shuffling
        presented in the file.
        """
        pfname = 'params' # Name of the parameter list file.
        
        dfname  = './' + set_name + '/train-data.csv'
        f = open(dfname, 'r')
        table = pd.read_csv(f)
        f.close()
        
        X_train = []
        y_train = []

        prefix = './'
        f = open(prefix + pfname + '.py', 'r')
        params = f.readlines()
        f.close()
        param_count = 0
        for param in params:
            if param[0] == '#':
                continue
            if '#' in param:
                param = param.split('#')[0]
            param_count += 1
            param = param[:len(param) - 1].strip()  # To get rid of the NewLine character at the end of the line.
            X_train.append( list( table[param] )  )
        
        X_train = np.matrix(X_train)
        X_train = X_train.transpose()
        X_train = extract_features.mat_norm(X_train)
        
        y_train = table['Labels']
        X_list  = list(X_train)

        return (X_train, y_train)

    @staticmethod
    def fetch_test():
        """
        Reads the file Test_data/left-lung-shuffled.csv and extracts it, separating
        the data from the labels. It takes no arguments, and uses the exact shuffling
        presented in the file.
        """
        pfname = 'params' # Name of the parameter list file.
        
        dfname  = './' + set_name + '/test-data.csv'
        f = open(dfname, 'r')
        table = pd.read_csv(f)
        f.close()
        
        X_test = []
        y_test = []

        prefix = './'
        f = open(prefix + pfname + '.py', 'r')
        params = f.readlines()
        f.close()
        param_count = 0
        for param in params:
            if param[0] == '#':
                continue
            if '#' in param:
                param = param.split('#')[0]
            param_count += 1
            param = param[:len(param) - 1].strip()  # To get rid of the NewLine character at the end of the line.
            X_test.append( list( table[param] )  )
        
        X_test = np.matrix(X_test)
        X_test = X_test.transpose()
        X_test = extract_features.mat_norm(X_test) #, a_mean = arg_mean, a_stdv = arg_stdv)

        y_test = table['Labels']
        
        return (X_test, y_test)
    
    @staticmethod
    def randomize(mat1, mat2):
        """
        Takes in two separate matrices, `mat1` and `mat2`, and fuses them into
        a single list of 2-tuples. These two tuples are of the form:
        
            ( <feature-vector>, <class-lable> )
            
        It then **permutes** this list using `random.shuffle`, and then returns the
        result.
        """
        list1 = []
        list2 = []
        
        for i in range(mat1.shape[0]):
            list1.append( (mat1[i,:], -1) )
        
        for i in range(mat2.shape[0]):
            list2.append( (mat2[i,:], 1) )
        
        overall = list1 + list2
        rnd.shuffle( overall )

        return overall

    @staticmethod
    def normalize(content, xm = None, ysd = None):
        """
        Takes in a list, normalizes it and returns it as a list.
        """
        X = np.array(content)
        #print 'DEBUG: ', X.shape

        if xm is None:
            x_mean = X.mean(axis = 0)
        else:
            x_mean = xm
        Y = X - x_mean

        if ysd is None:
            y_var = Y.var(axis = 0)
            y_sd  = np.sqrt( y_var )
        else:
            y_sd = ysd

        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                Y[i,j] = Y[i,j] / (y_sd[j] + 1e-10)
        
        return (x_mean, y_sd, list(Y))
    
    @staticmethod
    def separate(overall):
        """
        Takes in a list of 2-tuples, whose first members are 1-D matrices, and whose
        second members are always integers. The matrix indicates the feature vector,
        while the integer is 0 or 1 and represents the class.
        
        The output is a single 2-tuple of lists. The first is a list of feature vectors,
        which are in turn lists. The second member is a list of corresponding class
        numbers, which are integers.
        """
        X = []
        y = []
        for i in range( len(overall) ):
            y.append( overall[i][1] )
            U = overall[i][0]
            u = [U[0,j] for j in range(U.shape[1])]
            X.append(u)
        
        return (X, y)
