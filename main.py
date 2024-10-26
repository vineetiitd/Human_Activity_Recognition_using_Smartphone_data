# 1. Importing necesary Libraries
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# 2. Loading the Data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# 3.a Checking of duplicates
print("Number of duplicates in train : ", sum(train.duplicated()))
print("Number of duplicates in test : ", sum(test.duplicated()))

# 3.b Checking for missing values
print("Total number of missing values in train : ", train.isna().values.sum())
print("Total number of missing values in train : ", test.isna().values.sum())

# 3.c Checking for class imbalance
plt.figure(figsize=(10, 8))
plt.title("Barplot of Activity")
sns.countplot(train.Activity, order=train.Activity.value_counts().index)
plt.xticks(rotation=30)
plt.show()


# 4.a Analysing tBodyAccMag-mean feature
facetgrid = sns.FacetGrid(train, hue="Activity", height=5, aspect=3)
facetgrid.map(sns.distplot, "tBodyAccMag-mean()", hist=False).add_legend()

plt.annotate(
    "Static Activities",
    xy=(-0.98, 8),
    xytext=(-0.8, 16),
    arrowprops={"arrowstyle": "-", "ls": "dashed"},
)
plt.annotate(
    "Static Activities",
    xy=(-0.98, 13),
    xytext=(-0.8, 16),
    arrowprops={"arrowstyle": "-", "ls": "dashed"},
)
plt.annotate(
    "Static Activities",
    xy=(-0.98, 16),
    xytext=(-0.8, 16),
    arrowprops={"arrowstyle": "-", "ls": "dashed"},
)

plt.annotate(
    "Dynamic Activities",
    xy=(-0.2, 3.25),
    xytext=(0.1, 9),
    arrowprops={"arrowstyle": "-", "ls": "dashed"},
)
plt.annotate(
    "Dynamic Activities",
    xy=(0.1, 2.18),
    xytext=(0.1, 9),
    arrowprops={"arrowstyle": "-", "ls": "dashed"},
)
plt.annotate(
    "Dynamic Activities",
    xy=(-0.01, 2.15),
    xytext=(0.1, 9),
    arrowprops={"arrowstyle": "-", "ls": "dashed"},
)

plt.show()

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.title("Static Activities(closer view)")
sns.distplot(
    train[train["Activity"] == "SITTING"]["tBodyAccMag-mean()"],
    hist=False,
    label="Sitting",
)
sns.distplot(
    train[train["Activity"] == "STANDING"]["tBodyAccMag-mean()"],
    hist=False,
    label="Standing",
)
sns.distplot(
    train[train["Activity"] == "LAYING"]["tBodyAccMag-mean()"],
    hist=False,
    label="Laying",
)
plt.axis([-1.02, -0.5, 0, 17])

plt.subplot(1, 2, 2)
plt.title("Dynamic Activities(closer view)")
sns.distplot(
    train[train["Activity"] == "WALKING"]["tBodyAccMag-mean()"],
    hist=False,
    label="WALKING",
)
sns.distplot(
    train[train["Activity"] == "WALKING_DOWNSTAIRS"]["tBodyAccMag-mean()"],
    hist=False,
    label="WALKING_DOWNSTAIRS",
)
sns.distplot(
    train[train["Activity"] == "WALKING_UPSTAIRS"]["tBodyAccMag-mean()"],
    hist=False,
    label="WALKING_UPSTAIRS",
)
plt.show()

plt.figure(figsize=(10, 7))
sns.boxplot(x="Activity", y="tBodyAccMag-mean()", data=train, showfliers=False)
plt.ylabel("Body Acceleration Magnitude mean")
plt.title("Boxplot of tBodyAccMag-mean() column across various activities")
plt.axhline(y=-0.8, xmin=0.05, dashes=(3, 3))
plt.axhline(y=0.0, xmin=0.35, dashes=(3, 3))
plt.show()

# 4.b Visualizing data using PCA
x_for_pca = train.drop(["subject", "Activity"], axis=1)
pca = PCA(n_components=2, random_state=0).fit_transform(x_for_pca)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=pca[:, 0], y=pca[:, 1], hue=train["Activity"])
plt.show()

# 4.c Visualizing data using t-SNE
x_for_tsne = train.drop(["subject", "Activity"], axis=1)
tsne = TSNE(n_components=2, random_state=0, n_iter=1000).fit_transform(x_for_tsne)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=train["Activity"])
plt.show()


# 5. ML models
# Getting training and test data ready
X_train = train.drop(["subject", "Activity"], axis=1)
y_train = train.Activity

X_test = test.drop(["subject", "Activity"], axis=1)
y_test = test.Activity

# 5.a Logistic regression model with Hyperparameter tuning and cross validation
parameters = {"max_iter": [100, 200, 500]}
lr_classifier = LogisticRegression()
lr_classifier_rs = RandomizedSearchCV(
    lr_classifier, param_distributions=parameters, cv=5, random_state=42
)
lr_classifier_rs.fit(X_train, y_train)
y_pred_lr = lr_classifier_rs.predict(X_test)
lr_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_lr)
print("Accuracy using Logistic Regression : ", lr_accuracy)

# function to plot confusion matrix
def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(12, 8))  # for plotting confusion matrix as image
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.xticks(rotation=90)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()


cm = confusion_matrix(y_test.values, y_pred_lr)
plot_confusion_matrix(cm, np.unique(y_pred_lr))


# 5.b Kernel SVM model with Hyperparameter tuning and cross validation
parameters = {"kernel": ["linear", "rbf", "poly", "sigmoid"], "C": [100, 50]}

svm_rs = RandomizedSearchCV(
    SVC(), param_distributions=parameters, cv=3, random_state=42
)
svm_rs.fit(X_train, y_train)

kernel_svm_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy using Kernel SVM : ", kernel_svm_accuracy)

cm = confusion_matrix(y_test.values, y_pred)
plot_confusion_matrix(cm, np.unique(y_pred))

# 5.c Decision tree model with Hyperparameter tuning and cross validation
parameters = {"max_depth": np.arange(2, 10, 2)}

dt_classifier = DecisionTreeClassifier()
dt_classifier_rs = RandomizedSearchCV(
    dt_classifier, param_distributions=parameters, random_state=42
)
dt_classifier_rs.fit(X_train, y_train)

y_pred = dt_classifier_rs.predict(X_test)

dt_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy using Decision tree : ", dt_accuracy)

cm = confusion_matrix(y_test.values, y_pred)
plot_confusion_matrix(cm, np.unique(y_pred))

# 5.d Random forest model with Hyperparameter tuning and cross validation
parameters = {"n_estimators": np.arange(20, 101, 10), "max_depth": np.arange(2, 17, 2)}
rf_classifier = RandomForestClassifier()
rf_classifier_rs = RandomizedSearchCV(
    rf_classifier, param_distributions=parameters, random_state=42
)
rf_classifier_rs.fit(X_train, y_train)

get_best_randomsearch_results(rf_classifier_rs)

rf_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy using Random forest : ", rf_accuracy)

cm = confusion_matrix(y_test.values, y_pred)
plot_confusion_matrix(cm, np.unique(y_pred))
