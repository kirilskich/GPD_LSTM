import pandas as pd

days_to_predict = 4
test_data_size = 1
fut_pred = test_data_size + days_to_predict


#training params

epochs = 190
train_window = 20

#data

#df = pd.read_csv("data/gdp.csv")

df = pd.read_csv("data/gdp_q_usd.csv")