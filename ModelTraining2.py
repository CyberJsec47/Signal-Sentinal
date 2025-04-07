import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pickle


"""Model training, testing, and evaluation
    models used are:
    
    Random Forest
    Decision tree
    Logistic Regression
    SVM
    KNN
    Naive Bayes """

#features = ['RMS', 'Max Magnitude', 'Amplitude', 'PSD', 'Signal To Noise']

def randomForest():


    train_data = pd.read_csv('/home/josh/Documents/SignalSentinel/CSV_Files/train_data.csv')
    test_data = pd.read_csv('/home/josh/Documents/SignalSentinel/CSV_Files/test_data.csv')
    val_data = pd.read_csv('/home/josh/Documents/SignalSentinel/CSV_Files/val_data.csv')

    features = ['RMS', 'Max Magnitude', 'Amplitude', 'PSD', 'Signal To Noise']
    target = 'Classification'

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    X_val = val_data[features]
    y_val = val_data[target]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    model = RandomForestClassifier()

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5) 
    print("Cross-validation scores:", cv_scores)
    print("Average cross-validation score:", cv_scores.mean())

    model.fit(X_train_scaled, y_train)

    y_test_pred = model.predict(X_test_scaled)
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

    y_val_pred = model.predict(X_val_scaled)
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

    print("Confusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_test_pred))

    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

    print("Confusion Matrix (Validation Set):")
    print(confusion_matrix(y_val, y_val_pred))

    print("Classification Report (Validation Set):")
    print(classification_report(y_val, y_val_pred))



def decisionTree():

    train_data = pd.read_csv('train_scaled.csv')
    test_data = pd.read_csv('test_scaled.csv')
    val_data = pd.read_csv('val_scaled.csv')

    features = ['RMS', 'Max Magnitude', 'Amplitude', 'PSD', 'Signal To Noise']
    target = 'Classification'

    X_train = train_data[features]
    y_train = train_data[target]

    X_test = test_data[features]
    y_test = test_data[target]

    X_val = val_data[features]
    y_val = val_data[target]

    dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,                 
    min_samples_split=10,        
    min_samples_leaf=5,            
    class_weight='balanced'        
    )

    dt_model.fit(X_train, y_train)

    y_test_pred = dt_model.predict(X_test)

    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

    conf_matrix_test = confusion_matrix(y_test, y_test_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Safe', 'Jamming'], yticklabels=['Safe', 'Jamming'])
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    y_val_pred = dt_model.predict(X_val)

    print("Classification Report (Validation Set):")
    print(classification_report(y_val, y_val_pred))

    conf_matrix_val = confusion_matrix(y_val, y_val_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_val, annot=True, fmt='d', cmap='Blues', xticklabels=['Safe', 'Jamming'], yticklabels=['Safe', 'Jamming'])
    plt.title('Confusion Matrix (Validation Set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='accuracy')

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average cross-validation score: {cv_scores.mean()}")



def logisticRegression():

    train_data = pd.read_csv('/home/josh/Documents/SignalSentinel/CSV_Files/train_data.csv')
    test_data = pd.read_csv('/home/josh/Documents/SignalSentinel/CSV_Files/test_data.csv')
    val_data = pd.read_csv('/home/josh/Documents/SignalSentinel/CSV_Files/val_data.csv')

    features = ['RMS', 'Max Magnitude', 'Amplitude', 'PSD', 'Signal To Noise']
    target = 'Classification'

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    X_val = val_data[features]
    y_val = val_data[target]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  
    X_val_scaled = scaler.transform(X_val)  

    model = LogisticRegression(max_iter=1000)

    param_grid = {
        'C': [0.01, 0.1, 1, 10],  
        'penalty': ['l1', 'l2'],  
        'solver': ['liblinear'],
        'max_iter': [100, 200, 300]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=1)

    grid_search.fit(X_train_scaled, y_train)

    print("Best Hyperparameters from Grid Search:")
    print(grid_search.best_params_)
  
    best_model = grid_search.best_estimator_

    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Average cross-validation score:", cv_scores.mean())

    y_test_pred = best_model.predict(X_test_scaled)
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

    y_val_pred = best_model.predict(X_val_scaled)
    print("Validation Accuracy (Unseen Set):", accuracy_score(y_val, y_val_pred))

    print("Confusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_test_pred))

    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

    print("Confusion Matrix (Validation Set - Unseen):")
    print(confusion_matrix(y_val, y_val_pred))

    print("Classification Report (Validation Set - Unseen):")
    print(classification_report(y_val, y_val_pred))



def supportVectorMachine():

    train_data = pd.read_csv('/home/josh/Documents/SignalSentinel/CSV_Files/train_data.csv')
    test_data = pd.read_csv('/home/josh/Documents/SignalSentinel/CSV_Files/test_data.csv')
    val_data = pd.read_csv('/home/josh/Documents/SignalSentinel/CSV_Files/val_data.csv')

    features = ['RMS', 'Max Magnitude', 'Amplitude', 'PSD', 'Signal To Noise']
    target = 'Classification'

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    X_val = val_data[features]
    y_val = val_data[target]
 
    X_train = X_train.clip(lower=X_train.quantile(0.01), upper=X_train.quantile(0.99), axis=1)
    X_test = X_test.clip(lower=X_test.quantile(0.01), upper=X_test.quantile(0.99), axis=1)
    X_val = X_val.clip(lower=X_val.quantile(0.01), upper=X_val.quantile(0.99), axis=1)

    X_train['Signal To Noise'] = np.log1p(X_train['Signal To Noise'])
    X_test['Signal To Noise'] = np.log1p(X_test['Signal To Noise'])
    X_val['Signal To Noise'] = np.log1p(X_val['Signal To Noise'])

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    model = SVC(kernel='linear', C=0.1, max_iter=1000, tol=1e-5)

    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'gamma': ['scale', 'auto'],  
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=1)

    grid_search.fit(X_train_scaled, y_train)

    print("Best Hyperparameters from Grid Search:")
    print(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Average cross-validation score:", cv_scores.mean())

    y_test_pred = best_model.predict(X_test_scaled)
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

    y_val_pred = best_model.predict(X_val_scaled)
    print("Validation Accuracy (Unseen Set):", accuracy_score(y_val, y_val_pred))

    print("Confusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_test_pred))

    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

    print("Confusion Matrix (Validation Set - Unseen):")
    print(confusion_matrix(y_val, y_val_pred))



def knn():


    train_data = pd.read_csv('/home/josh/Documents/SignalSentinel/CSV_Files/train_data.csv')
    test_data = pd.read_csv('/home/josh/Documents/SignalSentinel/CSV_Files/test_data.csv')
    val_data = pd.read_csv('/home/josh/Documents/SignalSentinel/CSV_Files/val_data.csv')

    features = ['RMS', 'Max Magnitude', 'Amplitude', 'PSD', 'Signal To Noise']
    target = 'Classification'

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    X_val = val_data[features]
    y_val = val_data[target]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    knn = KNeighborsClassifier(n_neighbors=5)
    cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Average cross-validation score:", cv_scores.mean())

    knn.fit(X_train_scaled, y_train)

    y_test_pred = knn.predict(X_test_scaled)
    print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Confusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_test_pred))
    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

    y_val_pred = knn.predict(X_val_scaled)
    print("\nValidation Accuracy (Unseen Set):", accuracy_score(y_val, y_val_pred))
    print("Confusion Matrix (Validation Set - Unseen):")
    print(confusion_matrix(y_val, y_val_pred))
    print("Classification Report (Validation Set - Unseen):")
    print(classification_report(y_val, y_val_pred))


def naiveBayes():

    train_data = pd.read_csv('/home/josh/Documents/SignalSentinel/CSV_Files/train_data.csv')
    test_data = pd.read_csv('/home/josh/Documents/SignalSentinel/CSV_Files/test_data.csv')
    val_data = pd.read_csv('/home/josh/Documents/SignalSentinel/CSV_Files/val_data.csv')

    features = ['RMS', 'Max Magnitude', 'Amplitude', 'PSD', 'Signal To Noise']
    target = 'Classification'

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    X_val = val_data[features]
    y_val = val_data[target]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)

    y_test_pred = nb.predict(X_test_scaled)
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Confusion Matrix (Test Set):\n", confusion_matrix(y_test, y_test_pred))
    print("Classification Report (Test Set):\n", classification_report(y_test, y_test_pred))

    y_val_pred = nb.predict(X_val_scaled)
    print("Validation Accuracy (Unseen Set):", accuracy_score(y_val, y_val_pred))
    print("Confusion Matrix (Validation Set - Unseen):\n", confusion_matrix(y_val, y_val_pred))
    print("Classification Report (Validation Set - Unseen):\n", classification_report(y_val, y_val_pred))

    print(nb)

    try:
        with open('naive_bayes_model.pkl', 'wb') as model_file:
            pickle.dump(nb, model_file)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

    # After saving scaler
    try:
        with open('scaler.pkl', 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)
        print("Scaler saved successfully.")
    except Exception as e:
        print(f"Error saving scaler: {e}")

    print("Model and Scaler saved as naive_bayes_model.pkl and scaler.pkl")


naiveBayes()