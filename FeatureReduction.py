from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
import numpy as np
from sklearn.ensemble import RandomForestClassifier


"""Feature importance section of data preperation,
Firstly using random forest for feature importance, next T-SNE, ANOVA then PCA"""

def randomForest():

    df = pd.read_csv("train_scaled.csv")

    X_train = df.drop(["Classification", "Frequency", "Avg dBm", "Average Phase", "Entropy"], axis=1)  
    y_train = df["Classification"]

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    importance = model.feature_importances_
    features = X_train.columns

    plt.barh(features, importance)
    plt.title("Feature importance")
    plt.show()


def tsne_visuals():
  
    df = pd.read_csv("train_scaled.csv")


    X_train = df.drop(["Classification", "Frequency", "Avg dBm", "Average Phase", "Entropy"], axis=1) 
    y_train = df["Classification"]


    X = X_train.values
    y = y_train.values

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X)

    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y)
    plt.title("T-SNE Visualization")
    plt.show()


def anova():
 
    df = pd.read_csv("train_scaled.csv")

    X_train = df.drop(["Classification", "Frequency", "Avg dBm", "Average Phase", "Entropy"], axis=1) 
    y_train = df["Classification"]

    f_values, p_values = f_classif(X_train, y_train)

    feature_scores = pd.Series(f_values, index=X_train.columns).sort_values(ascending=False)
    
    print(feature_scores)


def pca_analysis():

    train_df = pd.read_csv('train_scaled.csv')
    test_df = pd.read_csv('test_scaled.csv')
    val_df = pd.read_csv('val_scaled.csv')

    X_train = train_df.drop(["Classification", "Frequency", "Avg dBm", "Average Phase", "Entropy"], axis=1) 
    y_train = train_df['Classification']

    X_test = test_df.drop(["Classification", "Frequency", "Avg dBm", "Average Phase", "Entropy"], axis=1)
    y_test = test_df['Classification']

    X_val = val_df.drop(["Classification", "Frequency", "Avg dBm", "Average Phase", "Entropy"], axis=1)
    y_val = val_df['Classification']

    pca = PCA(n_components=None)
    pca.fit(X_train)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o', linestyle='--')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.show()

    feature_names = X_train.columns
    components_df = pd.DataFrame(
        np.round(pca.components_, 3),
        columns=feature_names,
        index=[f'PC{i+1}' for i in range(len(feature_names))]
    )
    print(components_df)

    X_train_pca = pca.transform(X_train)[:, :2]


    pca_df = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2'])
    pca_df['Label'] = y_train.values

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Label', palette='coolwarm')
    plt.title('PCA of RF Signals (PC1 vs PC2)')
    plt.grid(True)
    plt.show()
