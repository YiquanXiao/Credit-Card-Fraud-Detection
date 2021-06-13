#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, classification_report, confusion_matrix, accuracy_score, precision_score, \
    recall_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import homogeneity_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler

from signal import signal, SIGPIPE, SIG_DFL

# Ignore SIG_PIPE and don't throw exceptions on it... (http://docs.python.org/library/signal.html)
signal(SIGPIPE, SIG_DFL)

# In[2]:


# Load the dataset from the csv file using pandas
data = pd.read_csv('creditcard.csv')
# Grab a peak at the data
data.head()

# In[3]:


# Dividing the X(features) and the Y(target) from the dataset
X = data.drop(["Class", "Time"], axis=1).values
Y = data["Class"].values
print(f'X shape: {X.shape}\nY shape: {Y.shape}')

# In[4]:


# Define the resampling method
resampling = SMOTE()
# Create the resampled feature set
X_resampled, Y_resampled = resampling.fit_sample(X, Y)

# In[11]:


# Create the training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=54321)
X_resampled_train, X_resampled_test, Y_resampled_train, Y_resampled_test = train_test_split(X_resampled, Y_resampled,
                                                                                            test_size=0.3,
                                                                                            random_state=54321)


# In[5]:


def evaluate(model_name, actual, prediction):
    print("the Model used is {}".format(model_name))
    acc = accuracy_score(actual, prediction)
    print("The accuracy is {}".format(acc))
    prec = precision_score(actual, prediction)
    print("The precision is {}".format(prec))
    rec = recall_score(actual, prediction)
    print("The recall is {}".format(rec))
    f1 = f1_score(actual, prediction)
    print("The F1-Score is {}".format(f1))
    mcc = matthews_corrcoef(actual, prediction)
    print("The Matthews correlation coefficient is {}".format(mcc))
    # Print the classifcation report and confusion matrix
    print("Classification report:\n", classification_report(actual, prediction))
    conf_mat = confusion_matrix(y_true=actual, y_pred=prediction)
    print("Confusion matrix:\n", conf_mat)


# In[6]:


def eval_roc(model, x_test, y_test):
    # Predict probabilities
    probs = model.predict_proba(x_test)
    # Print the ROC curve
    print('ROC Score:')
    print(roc_auc_score(y_test, probs[:, 1]))


# In[17]:


# Define the model as the random forest
# rf = RandomForestClassifier(random_state=5)
# rf.fit(X_resampled_train, Y_resampled_train)
# # predictions
# rf_predicted = rf.predict(X_resampled_test)
# evaluate("Random Forest", Y_resampled_test, rf_predicted)

# In[15]:


# Define the model as the random forest
# ann = MLPClassifier(random_state=0)
# ann.fit(X_resampled_train, Y_resampled_train)
# # predictions
# ann_predicted = ann.predict(X_resampled_test)
# evaluate("Artificial Neural Network", Y_resampled_test, ann_predicted)

# In[ ]:


print("Find Parameters for Random Forest...")


# In[14]:


# Define the parameter sets to test
params_rf = {'class_weight': ['balanced', 'balanced_subsample'],
             'n_estimators': [1, 10, 100],
             'max_features': ['auto', 'log2'],
             'max_depth': [4, 8, 10, 12, None],
             'criterion': ['gini', 'entropy']
             }

# Define the model to use
rf_base = RandomForestClassifier(random_state=5)

# Combine the parameter sets with the defined model
cv_rf = GridSearchCV(estimator=rf_base, param_grid=params_rf, cv=5, scoring='f1')

# Fit the model to our training data and obtain best parameters
cv_rf.fit(X_resampled_train, Y_resampled_train)
print("Best Parameters for RF:")
print(cv_rf.best_params_)
print("******************************")

# In[ ]:


# In[ ]:


print("Find Parameters for ANN...")


# In[16]:


# Define the parameter sets to test
params_ann = {'hidden_layer_sizes': [(100,), (100, 100), (200, 100)],
              'activation': ['logistic', 'relu'],
              'solver': ['adam', 'sgd']
              }

# Define the model as ANN
ann_base = MLPClassifier(random_state=0)

# Combine the parameter sets with the defined model
cv_ann = GridSearchCV(estimator=ann_base, param_grid=params_ann, cv=5, scoring='f1')

# Fit the model to our training data and obtain best parameters
cv_ann.fit(X_resampled_train, Y_resampled_train)
print("Best Parameters for ANN:")
print(cv_ann.best_params_)
print("******************************")

# In[ ]:


# In[ ]:


# Random Forest with Best Parameters
rf = RandomForestClassifier(class_weight=cv_rf.best_params_['class_weight'],
                            criterion=cv_rf.best_params_['criterion'],
                            max_depth=cv_rf.best_params_['max_depth'],
                            max_features=cv_rf.best_params_['max_features'],
                            n_estimators=cv_rf.best_params_['n_estimators'],
                            random_state=5)
rf.fit(X_resampled_train, Y_resampled_train)
# predictions
rf_predicted = rf.predict(X_resampled_test)
evaluate("Random Forest", Y_resampled_test, rf_predicted)
print("******************************")

# In[19]:


# ANN with Best params
ann = MLPClassifier(random_state=0,
                    hidden_layer_sizes=cv_ann.best_params_['hidden_layer_sizes'],
                    activation=cv_ann.best_params_['activation'],
                    solver=cv_ann.best_params_['solver'],
                    )
ann.fit(X_resampled_train, Y_resampled_train)
# predictions
ann_predicted = ann.predict(X_resampled_test)
evaluate("Artificial Neural Network", Y_resampled_test, ann_predicted)
print("******************************")

# In[ ]:
