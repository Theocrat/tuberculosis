from extract_features  import extract_features as ef
from validation_mc     import validation_mc    as val
from sklearn.ensemble  import RandomForestClassifier as rfc
from matplotlib.pyplot import stem, show

# PARAMETERS:
# ----------
num_est = 10         # --> num_est == Number of Estimators.
tree_depth = 7      # --> tree_depth == The depth of each of the trees.
seed = 0            # --> seed == The Random Seed of the tree.

# TRAINING PART:
# -------------

# Extracting Training Data:
(X_train, y_train) = ef.fetch_train()  # -->  Separates the data from the labels.
x_m, y_sd, X_train = ef.normalize(X_train)  # Normalizes the data, extracting mean and SD of training data for normalizing the testing data later.

clf = rfc(n_estimators = num_est, max_depth = tree_depth, random_state = seed)
clf.fit(X_train, y_train)

# Finding the Training Accuracy:
correct = 0
total   = 0

y_predict_train = clf.predict(X_train)
for i in range(len(y_predict_train)):
    if list(y_predict_train)[i] == list(y_train)[i]:
        correct += 1
    total += 1

print '\nTraining accuracy: ' + str(correct) + ' out of ' + str(total) +'.'
print 'Accuracy: ' + str( float(correct * 100) / total ) + '%.\n'


# ==================================

# TESTING PART:
# ------------

# Extracting Testing Data:
(X_test, y_test) = ef.fetch_test()
x_m, y_sd, X_test = ef.normalize(X_test, xm = x_m, ysd = y_sd)  # Normalizes the testing data with the mean and SD of the training set.

# Fitting the classifier:
print '\nUsing %s estimators of depth %s.\n' % (str(num_est), str(tree_depth))
    
# Testing the predictions:
correct = 0
total   = 0

y_hat = clf.predict(X_test)
#print 'Prediction\tActual'
for i in range(len(y_hat)):
    #print '', list(y_hat)[i], '-' *14, list(y_test)[i]
    if list(y_hat)[i] == list(y_test)[i]:
        correct += 1
    total += 1

print 'Testing Accuracy ' + str(correct) + ' out of ' + str(total) +'.'
print 'Accuracy: ' + str( float(correct * 100) / total ) + '%.'
