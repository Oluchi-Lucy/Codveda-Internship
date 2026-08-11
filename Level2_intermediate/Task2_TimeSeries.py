import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
df = pd.read_csv("2) Stock Prices Data Set.csv")

# Display the first five rows print(df.head())
# Display the first five rows print(df.columns())
# Display the first five rows print(df.info())

