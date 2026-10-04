# %%
import pandas as pd 
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv'
df3 = pd.read_csv(url)

# Task 1: Pakistan data
pk = df3[df3['Country/Region'] == 'Pakistan'].copy()

# Task 2: handle missing
pk['Province/State'] = pk['Province/State'].fillna('N/A')
pk[['Confirmed','Recovered','Deaths']] = pk[['Confirmed','Recovered','Deaths']].fillna(0)

# Task 3: line chart
pk['Date'] = pd.to_datetime(pk['Date'])
pk = pk.sort_values('Date')
plt.plot(pk['Date'], pk['Confirmed'])
plt.title('Pakistan COVID-19 Confirmed Cases Over Time')
plt.show()

# Task 4: day with highest NEW cases
pk['Daily_New'] = pk['Confirmed'].diff().fillna(0)
peak_idx = pk['Daily_New'].idxmax()
print("Highest new cases on:", pk.loc[peak_idx, 'Date'], "->", pk.loc[peak_idx, 'Daily_New'])

# Task 5: outliers
plt.boxplot(pk['Daily_New'])
plt.title('Daily New Cases - Outliers')
plt.show()
# %%
