import pandas as pd
data = pd.io.stata.read_stata('Fengs-stock-data.dta')
data.to_csv('Fengs-stock-data.csv')
print(data)