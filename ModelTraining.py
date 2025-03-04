import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


 
def test_1():

    """ This first test was using random forest and results
    shown perfect first time which is not ideal
    it uses a shuffled dataset to make sure the previosly used unshuffled wasnt negativily effecting it
    This model uses all features aviable including the freqency which when looking into a feature importance graph the RF classifer
    was using frequecy as it most important feature which for what this project is is no help as a safe and jamming signal can both be on the same frequency
    """

    df = pd.read_csv('Shuffled_dataset.csv')
    X = df.drop(columns=["Classification"])
    y = df["Classification"]

    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    scores = cross_val_score(model, X, y, cv=5)
    print("Cross-validation scores:", scores)
    print("Mean accuracy: ", scores.mean())

    feature_importance = model.feature_importances_
    features = X.columns

    plt.figure(figsize=(8, 5))
    plt.barh(features, feature_importance)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.show()

def test_2():

    """Test two is the same as the first but with removing frequency as a feature, same results perfect score but feature importance shows it looks
    more into other features for a decison
    next tests are to be made with other models to see perforamce changes
    """

    df = pd.read_csv("Shuffled_dataset.csv")

    X = df.drop(columns=["Frequency", "Classification"])
    y = df["Classification"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    feature_importance = model.feature_importances_

    feature_names = X.columns

    plt.figure(figsize=(10, 5))
    plt.barh(feature_names, feature_importance, color='blue')
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Names")
    plt.title("Feature importance removing Frequnecy")
    plt.show()


def test_3():

    """ Decision tree """

    df = pd.read_csv("Shuffled_dataset.csv")

    X = df.drop(columns=["Frequency", "Classification"])
    y = df["Classification"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)

    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Jamming"])
    disp.plot(cmap="Blues")
    plt.title("Decision Tree Confusion Matrix")
    plt.show()

    plt.figure(figsize=(15, 10))
    plot_tree(dt_model, feature_names=X.columns, class_names=["Safe", "Jamming"], filled=True)
    plt.title("Decision Tree Visuals")
    plt.show()

    cv_scores = cross_val_score(dt_model, X, y, cv=5)
    print(f"Cross validation scores: {cv_scores}")
    print(f"Mean Accruacy: {cv_scores.mean():.4f}")

    feature_importances = dt_model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_features = X.columns[sorted_indices]
    sorted_importances = feature_importances[sorted_indices]

    plt.figure(figsize=(10, 5))
    plt.barh(sorted_features, sorted_importances, color="blue")
    plt.xlabel("Feature Importnces")
    plt.ylabel("Features")
    plt.title("Decision tree feature importances")
    plt.gca().invert_yaxis()
    plt.show()

    """ Deicion tree used RMS as a sole feature for signal clasffication
    Maybe try hyperparemeter tuning but this seems like an unlikly model to use
    """


def test_4():

    """ KNN """

    df = pd.read_csv("Shuffled_dataset.csv")

    X = df.drop(columns=["Frequency", "Classification"])
    y = df["Classification"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_Scaled = scaler.fit_transform(X_train)
    X_test_Scaled = scaler.transform(X_test)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_Scaled, y_train)
    y_pred = knn_model.predict(X_test_Scaled)

    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Jamming"])
    disp.plot(cmap="Blues")
    plt.title("KNN Confusion MAtrix")
    plt.show()

    cv_scores = cross_val_score(knn_model, X, y, cv=5)
    print(f"Cross validation scores: {cv_scores}")
    print(f"Mean Accruacy: {cv_scores.mean():.4f}")


def test_5():

    """ SVM """
    df = pd.read_csv("Shuffled_dataset.csv")

    X = df.drop(columns=["Frequency", "Classification"])
    y = df["Classification"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_Scaled = scaler.fit_transform(X_train)
    X_test_Scaled = scaler.transform(X_test)

    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train_Scaled, y_train)

    y_pred = svm_model.predict(X_test_Scaled)

    print(classification_report(y_test, y_pred))
    cv_scores = cross_val_score(svm_model, X, y, cv=5)
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean Accuracy: {cv_scores.mean():.4f}")