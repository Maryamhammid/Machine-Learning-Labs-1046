# %%
import pandas as pd
import matplotlib.pyplot as plt

df_marks = pd.read_csv('../Datasets/students.csv')
df_attendance = pd.read_json('../Datasets/attendance.json')
df_bonus = pd.read_excel('../Datasets/extra.xlsx')

# Task 2: merge
merged = df_marks.merge(df_attendance, on='Name').merge(df_bonus, on='Name')
print(merged)

# Task 3: scatter, highlight attendance < 70%
low_att = merged[merged['Attendance'] < 70]
high_att = merged[merged['Attendance'] >= 70]
plt.scatter(high_att['Attendance'], high_att['Marks'], color='blue', label='>=70%')
plt.scatter(low_att['Attendance'], low_att['Marks'], color='red', label='<70%')
plt.xlabel('Attendance'); plt.ylabel('Marks'); plt.legend()
plt.show()

# Task 4: one-hot encode Name
merged_onehot = pd.get_dummies(merged, columns=['Name'], prefix='Name')
print(merged_onehot.head())
# %%
