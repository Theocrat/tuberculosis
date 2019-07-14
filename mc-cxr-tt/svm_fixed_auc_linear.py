from numpy             import array, trapz
from extract_features  import extract_features as ef
from validation_mc     import validation_mc    as val
from sklearn.svm       import SVC
from sklearn.metrics   import roc_curve, auc
import matplotlib.pyplot as plt

# PARAMETERS:
# ----------
num_est = 3         # --> num_est == Number of Estimators.
tree_depth = 3      # --> tree_depth == The depth of each of the trees.
seed = 0            # --> seed == The Random Seed of the tree.

# TRAINING PART:
# -------------

# Extracting Training Data:
(X_train, y_train) = ef.fetch_train()  # -->  Separates the data from the labels.
x_m, y_sd, X_train = ef.normalize(X_train)  # Normalizes the data, extracting mean and SD of training data for normalizing the testing data later.

clf = SVC(kernel = 'linear', C = 0.14, random_state = seed, probability = True)
clf.fit(X_train, y_train)

# Finding the Training Accuracy:
correct = 0
total   = 0

# ==================================

# TESTING PART:
# ------------

# Extracting Testing Data:
(X_test, y_test) = ef.fetch_test()
x_m, y_sd, X_test = ef.normalize(X_test, xm = x_m, ysd = y_sd)  # Normalizes the testing data with the mean and SD of the training set.

# Fitting the classifier:
print '\nUsing %s estimators of depth %s.\n' % (str(num_est), str(tree_depth))
    
# Deriving the ROC curve:
y_check = clf.predict(X_test)
y_hat = clf.predict_proba(X_test)
y_hat = array([ entry[1] for entry in y_hat ])
fpr, tpr, thresholds = roc_curve(y_test, y_hat, pos_label=1)

# Plotting the result:
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# Evaluating the AUC:
area = trapz(tpr, fpr)
print 'Area under the AUC curve:', area
