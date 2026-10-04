
# Handling Missing Values
# %%
import pandas as pd
data = {'A': [1, 2, None, 4, 5], 'B': [6, None, 8, 9, 10], 'C': [11, 12, 13, None, 15]}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

df_dropna_rows = df.dropna(axis=0)
df_dropna_columns = df.dropna(axis=1)
df_fill_median = df.fillna(df.median())

print("\nAfter dropping rows with missing values:")
print(df_dropna_rows)
print("\nAfter dropping columns with missing values:")
print(df_dropna_columns)
print("\nAfter filling missing values with median:")
print(df_fill_median)


# %%
from sklearn.preprocessing import LabelEncoder

# Manual mapping
mapping = {'h': 1, 'u': 2, 't': 3}
features = pd.DataFrame({'Type': ['h', 'u', 't', 'h', 'u']})
features['Type'] = features['Type'].map(mapping)
print("Encoded Types:")
print(features.groupby('Type').size())

# Label Encoding
features_df = pd.DataFrame({'Regionname': ['North', 'South', 'East', 'West', 'North']})
le = LabelEncoder()
features_df['Region'] = le.fit_transform(features_df['Regionname'])
print("\nLabel Encoded Regions:")
print(features_df.value_counts())

# One-Hot Encoding
features2 = pd.DataFrame({'Method': ['Method1', 'Method2', 'Method1', 'Method3', 'Method2']})
df_one_hot = pd.get_dummies(features2['Method'])
print("\nOne Hot Encoded DataFrame:")
print(df_one_hot.value_counts())


# Ordinal Encoding

# %%
from sklearn.preprocessing import OrdinalEncoder

edu = pd.DataFrame({'Education': ['High School', 'Bachelors', 'Masters', 'PhD', 'Bachelors']})
order = [['High School', 'Bachelors', 'Masters', 'PhD']]
oe = OrdinalEncoder(categories=order)
edu['Education_Encoded'] = oe.fit_transform(edu[['Education']])
print(edu)

# Handling Outliers (boxplot + IQR)
# %%
import matplotlib.pyplot as plt

sample = [10, 12, 11, 13, 12, 14, 13, 100]
plt.boxplot(sample)
plt.title("Outlier Visualization")
plt.show()

# %%
income = pd.DataFrame({'Income': [25000, 27000, 26000, 28000, 500000, 24000]})

Q1 = income['Income'].quantile(0.25)
Q3 = income['Income'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = income[(income['Income'] < lower_limit) | (income['Income'] > upper_limit)]
print(outliers)

data_cleaned = income[(income['Income'] >= lower_limit) & (income['Income'] <= upper_limit)]
print(data_cleaned)


# %%
import pandas as pd
df=pd.read_csv("../Data Set/Admission_Predict.csv")
coulumns=df.columns
df.drop("Serial#",axis=1,inplace=True)
y = df['Chance of Admit']
df.drop("Chance of Admit",axis=1,inplace=True)
df.head()

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Initialize and train the linear regression model
simple_lr = LinearRegression()
simple_lr.fit(df[['GRE Score']], y)
plt.scatter(df['GRE Score'], y, color='green', label='Actual Data', alpha=0.5)
plt.plot(df['GRE Score'], simple_lr.predict(df[['GRE Score']]), color='red', linewidth=3, label='Regression Line')
plt.xlabel('GRE Score')
plt.ylabel('Admission chance')
plt.title('GRE Score vs Admission chance')
plt.legend()
plt.show()


# %%
