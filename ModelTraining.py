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
    df = pd.read_csv("preprocessed_data.csv")

    features = ['Avg dBm', 'PSD', 'Entropy', 'RMS'] 
    X_selected = df[features]
    y = df['Classification'] 

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_selected)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    scores = cross_val_score(model, X_selected,  y, cv=5)
    print("Cross-validation scores:", scores)
    print("Mean accuracy: ", scores.mean())

    feature_importance = model.feature_importances_

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

    df = pd.read_csv("preprocessed_data.csv")

    features = ['Avg dBm', 'PSD', 'Entropy', 'RMS'] 
    X_selected = df[features]
    y = df['Classification'] 

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_selected)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    feature_importance = model.feature_importances_

    feature_names = X_selected

    plt.figure(figsize=(10, 5))
    plt.barh(feature_names, feature_importance, color='blue')
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Names")
    plt.title("Feature importance removing Frequnecy")
    plt.show()


def test_3():

    """ Decision tree """

    df = pd.read_csv("preprocessed_data.csv")

    features = ['Avg dBm', 'PSD', 'Entropy', 'RMS'] 
    X_selected = df[features]
    y = df['Classification'] 

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_selected)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
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
    plot_tree(dt_model, feature_names=X_selected, class_names=["Safe", "Jamming"], filled=True)
    plt.title("Decision Tree Visuals")
    plt.show()

    cv_scores = cross_val_score(dt_model, X_selected, y, cv=5)
    print(f"Cross validation scores: {cv_scores}")
    print(f"Mean Accruacy: {cv_scores.mean():.4f}")

    feature_importances = dt_model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_features = X_selected[sorted_indices]
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
    """ KNN with Min-Max Scaling """

    data = pd.read_csv("preprocessed_data.csv")

    X = data[['Avg dBm', 'PSD', 'Entropy', 'RMS']]  
    y = data['Classification']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(conf_matrix)

    with open("knn.pkl", "wb") as model_file:
        pickle.dump(knn, model_file)
    with open("KNN_scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    print("Model and scaler saved.")
    

def test_5():
    """Train, Standardize, and Export SVM Model"""

    data = pd.read_csv("preprocessed_data.csv")

    X = data[['Avg dBm', 'PSD', 'Entropy', 'RMS']]  
    y = data['Classification'] 

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


def preprocess_rf_dataset(df, test_size=0.2, random_state=42, save_path=None):
 
    df = df.drop(columns=["Frequency"], errors="ignore")

    X = df.drop(columns=["Classification"])
    y = df["Classification"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    df_preprocessed = X_scaled.copy()
    df_preprocessed["Classification"] = y_encoded

    if save_path:
        df_preprocessed.to_csv(save_path, index=False)
        print(f"Preprocessed dataset saved as: {save_path}")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, 
                                                        test_size=test_size, 
                                                        random_state=random_state, 
                                                        stratify=y_encoded)

    return X_train, X_test, y_train, y_test


def split_and_encode_csv(input_csv, output_train='train_data.csv', output_test='test_data.csv', output_val='val_data.csv'):
    # Read the input CSV file
    df = pd.read_csv(input_csv)
    
    # Encode the 'Classification' column
    encoder = LabelEncoder()
    df['Classification'] = encoder.fit_transform(df['Classification'])
    
    # Split the data into train, test, and validation sets
    train_data, temp_data = train_test_split(df, test_size=0.4, random_state=42)  # 60% train, 40% temp (split into test/val)
    test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)  # Split remaining into 50% test and 50% val
    
    # Save the splits into CSV files
    train_data.to_csv(output_train, index=False)
    test_data.to_csv(output_test, index=False)
    val_data.to_csv(output_val, index=False)



def stanardScaler():
    # Load your datasets
    train_data = pd.read_csv('train_data.csv')
    test_data = pd.read_csv('test_data.csv')
    val_data = pd.read_csv('val_data.csv')

    # Separate the features (X) and the target variable (y) - "Classification" column
    X_train = train_data.drop(columns=['Classification'])
    y_train = train_data['Classification']

    X_test = test_data.drop(columns=['Classification'])
    y_test = test_data['Classification']

    X_val = val_data.drop(columns=['Classification'])
    y_val = val_data['Classification']

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform the test and validation data using the same scaler
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    # Convert the scaled data back to DataFrame (optional: specify column names if necessary)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)

    # Add the "Classification" column back to the scaled data
    train_scaled_df = pd.concat([X_train_scaled_df, y_train], axis=1)
    test_scaled_df = pd.concat([X_test_scaled_df, y_test], axis=1)
    val_scaled_df = pd.concat([X_val_scaled_df, y_val], axis=1)

    # Save the scaled datasets to new CSV files
    train_scaled_df.to_csv('train_scaled.csv', index=False)
    test_scaled_df.to_csv('test_scaled.csv', index=False)
    val_scaled_df.to_csv('val_scaled.csv', index=False)

    print("Scaling applied to features and new files saved.")
