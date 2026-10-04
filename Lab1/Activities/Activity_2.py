# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
np.random.seed(1)
n = 100
df2 = pd.DataFrame({
    'ID': range(1, n+1),
    'Math': np.random.randint(40, 100, n).astype(float),
    'Science': np.random.randint(40, 100, n).astype(float),
    'English': np.random.randint(40, 100, n).astype(float),
    'Grade': np.random.choice(['A', 'B', 'C'], n, p=[0.3, 0.4, 0.3])
})
missing_idx = np.random.choice(df2.index, 8, replace=False)
df2.loc[missing_idx[:3], 'Math'] = np.nan
df2.loc[missing_idx[3:6], 'Science'] = np.nan
df2.loc[missing_idx[6:], 'English'] = np.nan

# Task 1: fill missing with mean
for col in ['Math', 'Science', 'English']:
    df2[col] = df2[col].fillna(df2[col].mean())

# Task 2: encode Grade
le2 = LabelEncoder()
df2['Grade_Encoded'] = le2.fit_transform(df2['Grade'])

# Task 3: histogram
plt.hist(df2['Math'], bins=10)
plt.title('Math Scores Distribution')
plt.show()

# Task 4: boxplot on total score
df2['Total'] = df2['Math'] + df2['Science'] + df2['English']
plt.boxplot(df2['Total'])
plt.title('Total Score Boxplot')
plt.show()

# Task 5: summary
summary = df2.groupby('Grade')[['Math','Science','English','Total']].mean().sort_values('Total', ascending=False)
print(summary)
print("Best performing grade overall:", summary.index[0])
# %%
