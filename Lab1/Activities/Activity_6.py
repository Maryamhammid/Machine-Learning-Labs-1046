# %%

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Task 1: Explore dataset (jaisa Iris wale example mein tha)
cancer = load_breast_cancer()
print(cancer.keys())
print(cancer.target_names)
n_samples, n_features = cancer.data.shape
print('Number of samples:', n_samples)
print('Number of features:', n_features)
print('Feature Names:', cancer.feature_names)
print("Dimension of Input", cancer.data.shape)
print("Dimension of Output", cancer.target.shape)

df6 = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df6['target'] = cancer.target


# Task 2: Missing values 
missing_rows = np.random.choice(df6.index, 5, replace=False)
df6.loc[missing_rows, 'mean radius'] = np.nan
print(df6['mean radius'].isnull().sum())

df6['mean radius'] = df6['mean radius'].fillna(df6['mean radius'].median())
print(df6['mean radius'].isnull().sum())


# Task 3: Class distribution + SMOTE 
print(df6['target'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    df6.drop(columns='target'), df6['target'],
    test_size=0.2, random_state=42, stratify=df6['target']
)
X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)
print(y_train.value_counts())
print(y_bal.value_counts())


# Task 4: Outliers - IQR method
Q1 = df6['mean radius'].quantile(0.25)
Q3 = df6['mean radius'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df6[(df6['mean radius'] < lower_limit) | (df6['mean radius'] > upper_limit)]
print(outliers)

plt.boxplot([df6['mean radius'], df6['mean texture']])
plt.show()


# Task 5: Log transformation
df6['mean area log'] = np.log(df6['mean area'])

plt.hist(df6['mean area'], bins=30, alpha=0.5, label='before')
plt.hist(df6['mean area log'], bins=30, alpha=0.5, label='after log')
plt.legend()
plt.show()


# Task 6: Feature Scaling 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df6[cancer.feature_names])


# Task 7: PCA + scatter 
X_pca = PCA(n_components=2).fit_transform(X_scaled)
plt.scatter(X_pca[df6['target']==0, 0], X_pca[df6['target']==0, 1], label='malignant')
plt.scatter(X_pca[df6['target']==1, 0], X_pca[df6['target']==1, 1], label='benign')
plt.legend()
plt.show()


# Task 8: Train 2 classifiers, compare
Xtr, Xte, ytr, yte = train_test_split(X_scaled, df6['target'], test_size=0.2, random_state=42, stratify=df6['target'])
svm = SVC().fit(Xtr, ytr)
knn = KNeighborsClassifier().fit(Xtr, ytr)
print("SVM accuracy:", accuracy_score(yte, svm.predict(Xte)))
print("KNN accuracy:", accuracy_score(yte, knn.predict(Xte)))
# %%
