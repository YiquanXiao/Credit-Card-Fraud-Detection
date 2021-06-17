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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, classification_report, confusion_matrix, accuracy_score, precision_score, \
    recall_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import homogeneity_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from itertools import product

# Load the dataset from the csv file using pandas
data = pd.read_csv('creditcard.csv')

# Define the scaler and apply to the data
scaler = StandardScaler()
temp = data['Amount']
data['Amount'] = scaler.fit_transform(temp.values.reshape(-1, 1))

# Dividing the X(features) and the Y(target) from the dataset
X = data.drop(["Class", "Time"], axis=1).values
Y = data["Class"].values
print(f'X shape: {X.shape}\nY shape: {Y.shape}')

# Create the training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Define the resampling method
resampling = SMOTE()
# Create the resampled feature set
X_resampled_train, Y_resampled_train = resampling.fit_sample(X_train, Y_train)


def plot_confusion_matrix(cm, classes=None,
                          normalize=False,
                          title='Fraud Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    From:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-
        examples-model-selection-plot-confusion-matrix-py
    """
    if classes is None:
        classes = ['Not Fraud', 'Fraud']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


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
    plot_confusion_matrix(conf_mat)


def eval_roc(model, x_test, y_test):
    # Predict probabilities
    probs = model.predict_proba(x_test)
    # Print the ROC curve
    print('ROC Score:')
    print(roc_auc_score(y_test, probs[:, 1]))


def tune_ann(hidden_layers, activations, solvers, learning_rates, learning_rate_inits):
    best_h = None
    best_a = None
    best_s = None
    best_lr = None
    best_lri = None
    best_recall = 0
    best_accuracy = 0
    best_predicted = None
    for h in hidden_layers:
        for a in activations:
            for s in solvers:
                for lr in learning_rates:
                    for lri in learning_rate_inits:
                        model = MLPClassifier(random_state=0, hidden_layer_sizes=h, activation=a, solver=s,
                                              learning_rate=lr, learning_rate_init=lri)
                        model.fit(X_resampled_train, Y_resampled_train)
                        model_predicted = model.predict(X_test)
                        rec = recall_score(Y_test, model_predicted)
                        acc = accuracy_score(Y_test, model_predicted)
                        if rec > best_recall:
                            best_h = h
                            best_a = a
                            best_s = s
                            best_lr = lr
                            best_lri = lri
                            best_recall = rec
                            best_accuracy = acc
                            best_predicted = model_predicted
                        elif rec == best_recall and acc > best_accuracy:
                            best_h = h
                            best_a = a
                            best_s = s
                            best_lr = lr
                            best_lri = lri
                            best_recall = rec
                            best_accuracy = acc
                            best_predicted = model_predicted
    print("Best hidden layer:", best_h)
    print("Best activation:", best_a)
    print("Best solver:", best_s)
    print("Best learning_rate:", best_lr)
    print("Best learning_rate_init:", best_lri)
    print("Corresponding Recall:", best_recall)
    print("Corresponding Accuracy:", best_accuracy)
    conf_mat = confusion_matrix(y_true=Y_test, y_pred=best_predicted)
    print("Confusion matrix:\n", conf_mat)
    plot_confusion_matrix(conf_mat)


tune_ann(hidden_layers=[(250,), (100,), (100, 100)],
         activations=['relu', 'logistic'],
         solvers=['adam', 'sgd'],
         learning_rates=['constant', 'adaptive'],
         learning_rate_inits=[0.001, 0.01, 0.1])
