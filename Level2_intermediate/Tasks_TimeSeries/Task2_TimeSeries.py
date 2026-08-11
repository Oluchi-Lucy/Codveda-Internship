import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
df = pd.read_csv("2) Stock Prices Data Set.csv")
# Display the first five rows
print(df.head())
# Display the column names
print(df.columns)
# Display the info about the dataset
print(df.info())
print(df['symbol'].unique())
df_single = df[df['symbol'] == 'AAPL']
print(df_single.head())
print(df_single.shape)
df_single['date'] = pd.to_datetime(df_single['date'])
print(df_single.dtypes)
df_single = df_single.sort_values('date')
df_single = df_single.set_index('date')
print(df_single.head())
plt.figure(figsize=(12, 5))
plt.plot(df_single.index, df_single['close'])
plt.title("AAPL Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df_single['close'], model='additive', period=30)
decomposition.plot()
plt.savefig("aapl_seasonal_decomposition.png")
plt.show()
df_single['moving_avg_30'] = df_single['close'].rolling(window=30).mean()

plt.figure(figsize=(12, 5))
plt.plot(df_single.index, df_single['close'], label='Close Price', alpha=0.5)
plt.plot(df_single.index, df_single['moving_avg_30'], label='30-Day Moving Average', color='red')
plt.title("AAPL Close Price with 30-Day Moving Average")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.savefig("aapl_moving_average.png")
plt.show()
