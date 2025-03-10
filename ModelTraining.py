import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import pickle



def min_max_scaler():

    """Function to proccess data as min max scaling and standarization"""

    df = pd.read_csv("Shuffled_dataset.csv")
    print(df.columns.tolist())

    feature_columns = ['Frequency','Signal To Noise','Max Magnitude','Avg dBm','Average Phase','Entropy','PSD','Amplitude','RMS']
    X = df[feature_columns]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    scaler_std = StandardScaler()
    X_standardized = scaler_std.fit_transform(X)

    df[feature_columns] = X_scaled


def visualise_datasets():

    """ Visualisation of the three datasets from the standard training set, scaled and standardized. this is hopfully going to help model
    training to produce reliable results as all tests so far are 100% accurate"""

    original_df = pd.read_csv("Shuffled_dataset.csv")
    minmax_df = pd.read_csv("Dataset_scaled.csv")
    standardized_df = pd.read_csv("Dataset_standard.csv")

    label_column = "Classification"
    if label_column in original_df.columns:
        original_df = original_df.drop(columns=[label_column])
    if label_column in minmax_df.columns:
        minmax_df = minmax_df.drop(columns=[label_column])
    if label_column in standardized_df.columns:
        standardized_df = standardized_df.drop(columns=[label_column])

    features = ['Frequency', 'Signal To Noise', 'Max Magnitude', 'Avg dBm', 'Average Phase',
                'Entropy', 'PSD', 'Amplitude', 'RMS']

    summary_df = pd.DataFrame({
        "Dataset": ["Original", "Min-Max Scaled", "Standardized"],
        "Mean": [original_df.mean().mean(), minmax_df.mean().mean(), standardized_df.mean().mean()],    
        "Std Dev": [original_df.std().mean(), minmax_df.std().mean(), standardized_df.std().mean()]
    })

    print(summary_df)


def test_1():
    # Random forest 

    """ This first test was using random forest and results
    shown perfect first time which is not ideal
    it uses a shuffled dataset to make sure the previosly used unshuffled wasnt negativily effecting it
    This model uses all features aviable including the freqency which when looking into a feature importance graph the RF classifer
    was using frequecy as it most important feature which for what this project is is no help as a safe and jamming signal can both be on the same frequency
    """

    df = pd.read_csv('Dataset_standard.csv')
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

    # Random forest 

    """Test two is the same as the first but with removing frequency as a feature, same results perfect score but feature importance shows it looks
    more into other features for a decison
    next tests are to be made with other models to see perforamce changes
    """

    df = pd.read_csv("Dataset_standardized.csv")

    X = df.drop(columns=["Classification"])
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

    df = pd.read_csv("Dataset_standardized.csv")

    X = df.drop(columns=["Classification"])
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

    
    df = pd.read_csv("Dataset_standardized.csv")
    X = df.drop(columns=["Classification"])
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
    #plt.show()
    print("Confusion Matrix:\n", cm)


    print("Class distribution in y_train:")
    print(y_train.value_counts())

    print("Class distribution in y_test:")
    print(y_test.value_counts())

    corr_matrix = X.corr()
    print(corr_matrix)


    cv_scores = cross_val_score(knn_model, X_train_Scaled, y_train, cv=5)
    print(f"Cross-validation scores (scaled data): {cv_scores}")
    print(f"Mean Cross-validation Accuracy: {cv_scores.mean():.4f}")

    precision = precision_score(y_test, y_pred, pos_label="Jamming", average="binary")
    recall = recall_score(y_test, y_pred, pos_label="Jamming", average="binary")
    f1 = f1_score(y_test, y_pred, pos_label="Jamming", average="binary")
    auc = roc_auc_score(y_test, knn_model.predict_proba(X_test_Scaled)[:, 1])

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    strat_kfold = StratifiedKFold(n_splits=5)
    strat_cv_scores = cross_val_score(knn_model, X_train_Scaled, y_train, cv=strat_kfold)
    print(f"Stratified Cross-validation scores: {strat_cv_scores}")
    print(f"Mean Stratified Cross-validation Accuracy: {strat_cv_scores.mean():.4f}")


def test_5():
    """Train, Standardize, and Export SVM Model"""

    df = pd.read_csv("Dataset_standardized.csv")

    X = df.drop(columns=["Classification"])
    y = df["Classification"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    cv_scores = cross_val_score(svm_model, X_scaled, y, cv=5)
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean Accuracy: {cv_scores.mean():.4f}")

    with open("svm_jamming_detector.pkl", "wb") as model_file:
        pickle.dump(svm_model, model_file)
    
    with open("scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    print("Model and scaler saved.")



def test_6():

    """ Logistic regression"""

    df = pd.read_csv("Dataset_standardized.csv")
    
    
    X = df.drop(columns=["Classification"])
    y = df["Classification"]


    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=10000, random_state=0)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(X_train.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.show()

    y_prob = clf.predict_proba(X_test)[:, 1] 
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--') 
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


def test_7():

    """Naive Bayes"""

    df = pd.read_csv("Dataset_standardized.csv")
    X = df.drop(columns=["Classification"]) 
    y = df["Classification"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    y_pred = nb_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    print(f"Naive Bayes Model Accuracy: {acc:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    y_prob = nb_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Naive Bayes')
    plt.legend(loc='lower right')
    plt.show()

