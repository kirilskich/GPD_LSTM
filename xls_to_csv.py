import pandas as pd

data = pd.read_excel('data/gdp.xls', index_col=None)
data.to_csv ('data/gdp.csv', index = None, header=True)