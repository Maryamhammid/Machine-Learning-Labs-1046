# %%

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd

data5 = {
    "Name": ["Ahsan", "Hira", "Bilal", "Zara", "Salman", "Mahnoor"],
    "Age": [25, 27, 35, 29, None, 40],
    "Salary": [50000, None, 75000, 2000000, 60000, 90000],
    "Department": ["IT", "Finance", "IT", "HR", "Finance", "IT"]
}
df5 = pd.DataFrame(data5)

# Task 1: missing values -> median (mean use na karo, salary mein already ek bohat bada outlier hai)
df5['Age'] = df5['Age'].fillna(df5['Age'].median())
df5['Salary'] = df5['Salary'].fillna(df5['Salary'].median())

# Task 2: encode Department
le5 = LabelEncoder()
df5['Department_Encoded'] = le5.fit_transform(df5['Department'])
print(df5)

# Task 3: outliers
plt.boxplot(df5['Salary'])
plt.title('Salary Outlier Detection')
plt.show()

Q1, Q3 = df5['Salary'].quantile(0.25), df5['Salary'].quantile(0.75)
IQR = Q3 - Q1
upper = Q3 + 1.5*IQR

# Task 4: Zara's comment
zara_salary = df5.loc[df5['Name']=='Zara','Salary'].values[0]
print("Zara outlier?", zara_salary > upper)
# %%
