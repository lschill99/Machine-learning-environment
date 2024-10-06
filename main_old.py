import numpy as np



import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, make_scorer, precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from skopt.space import Real, Integer


import scipy.io
import pandas as pd

from model import Model

TEST_SIZE = 0.2
VALIDATION_SIZE= 0.2
TRAIN_SIZE= 0.6

RANDOMSTATE = 42
N_SPLITS = 3

def load_data(path):
    file_path = path
    # Load MATLAB file if(path extists)
    mat = scipy.io.loadmat(file_path)
    print(mat)
    X = mat['X']
    X_dense = X.todense()
    Y = mat['Y'].ravel()
    
    df_X = pd.DataFrame(X_dense)
    df_Y = pd.DataFrame(Y)
    return df_X, df_Y

# Define the file path
file_path = 'C:/Users/lasse/OneDrive/Dokumente/MASTER/Semester SS24/ML1/Abschlussprojekt_EMAIL/emails'



X , y = load_data('data/emails.mat')

# First, split into train/validatio and remaining data (tes)
X_train_val, X_test, y_train_val, y_test = train_test_split(X.T, y, test_size=TEST_SIZE) #, random_state=RANDOMSTATE

# Now split the train/validation data into validation and train sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125) #, random_state=RANDOMSTATE

# Result:
# - X_train, y_train for training
# - X_val, y_val for validation
# - X_test, y_test for final testing

def make_precision_score_for_ham(y_true, y_pred):
    return precision_score(y_true,y_pred, labels=[-1], zero_division=0)

random_forest = Model(RandomForestClassifier()) 
random_forest.model.fit(X_train,y_train)
predictions = random_forest.model.predict(X_train)
precision_score_for_ham = make_precision_score_for_ham(predictions, y_train) 

param_space = {
    'n_estimators': Integer(50, 500),              # Number of trees
    'max_depth': Integer(2, 30),                   # Maximum depth of the trees
    'min_samples_split': Integer(2, 10),           # Minimum number of samples to split a node
    'min_samples_leaf': Integer(1, 10),            # Minimum samples per leaf
    #'max_features': ['auto'],      # Number of features considered for splits was ['auto', 'sqrt', 'log2']
    'bootstrap': [False]                     # Whether to use bootstrap sampling was TRUE,FALSE
}





# Step 1: Define the cross-validation strategy (n-fold, in this case 3-fold)
cv_strategy = KFold(n_splits=N_SPLITS, shuffle=True, random_state= RANDOMSTATE)



# Step 4: Initialize BayesSearchCV
bayes_search = BayesSearchCV(
    estimator=random_forest.model,            # The model (random forest in this case)
    search_spaces=param_space,  # The search space for hyperparameters
    n_iter=10,               # Number of iterations to run (adjust as needed)
    cv=cv_strategy,          # Use 5-fold cross-validation
    n_jobs= 1,               # Use all available CPU cores
    random_state= RANDOMSTATE,         # For reproducibility
    scoring= make_scorer(precision_score, pos_label=-1)
)

# Step 2: Fit the model and perform hyperparameter tuning
bayes_search.fit(X_val, y_val)


# Step 3: Print the best hyperparameters found
print("Best hyperparameters found:", bayes_search.best_params_)

# Step 4: Print the best cross-validation score
print("Best cross-validation score:", bayes_search.best_score_)

