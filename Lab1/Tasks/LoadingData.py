# %%
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['Lahore', 'Faisalabad', 'Karachi', 'Multan'],
    'Salary': [50000, 60000, 70000, 55000]
}
df_manual = pd.DataFrame(data)
print("Manual DataFrame:")
print(df_manual)

# %%

# LOADING FROM CSV FILE
# %%
import pandas as pd

df_csv = pd.read_csv('../Datasets/cars.csv') 
print("CSV Data:")
print(df_csv.head())

online_csv_url = 'https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv'
df_online_csv = pd.read_csv(online_csv_url)
print("\nOnline CSV Data:")
print(df_online_csv.head())

# LOADING FROM JSON FILE
# %%
df_json = pd.read_json('../Datasets/attendance.json')   # <-- apni JSON file ka naam yahan
print("JSON Data:")
print(df_json.head())

# LOADING FROM EXCEL FILE
# %%
df_excel = pd.read_excel('../Datasets/extra.xlsx')   # <-- apni Excel file ka naam yahan
print("Excel Data:")
print(df_excel.head())


# LOADING FROM TEXT FILE
# %%
df_text = pd.read_csv('../Datasets/data.txt', delimiter='\t')   # tab-separated file
print("Text File Data:")
print(df_text.head())