# %%
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = {
    'Province': ['Punjab', 'Sindh', 'KPK', 'Balochistan'],
    'Population_millions': [110.0, 47.9, 35.5, None],
    'Literacy_Rate': [61.0, None, 53.0, 41.0],
    'Region': ['North', 'South', 'North', 'West']
}
df1 = pd.DataFrame(data)

# Task 1: missing values
df1['Literacy_Rate'] = df1['Literacy_Rate'].fillna(df1['Literacy_Rate'].median())
df1_clean = df1.dropna(subset=['Population_millions']).copy()
print(df1_clean)

# Task 2: encode Region
le = LabelEncoder()
df1_clean['Region_Label'] = le.fit_transform(df1_clean['Region'])
df1_onehot = pd.get_dummies(df1_clean, columns=['Region'], prefix='Region')
print(df1_clean)
print(df1_onehot)

# Task 3: scatter plot
plt.scatter(df1_clean['Population_millions'], df1_clean['Literacy_Rate'])
for _, row in df1_clean.iterrows():
    plt.annotate(row['Province'], (row['Population_millions'], row['Literacy_Rate']))
plt.xlabel('Population (millions)'); plt.ylabel('Literacy Rate (%)'); plt.title('Population vs Literacy Rate')
plt.show()

# Task 4: outlier detection
Q1, Q3 = df1['Literacy_Rate'].quantile(0.25), df1['Literacy_Rate'].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
print(df1[(df1['Literacy_Rate'] < lower) | (df1['Literacy_Rate'] > upper)])
# %%
