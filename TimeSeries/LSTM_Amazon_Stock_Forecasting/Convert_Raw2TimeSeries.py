import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
sys.stdout = open("DataFrameRecording.txt", "w", encoding="utf-8")


data = pd.read_csv('AMZN.csv')
data['Date'] = pd.to_datetime(data['Date'])
print(data)
data = data[['Date', 'Close']]
print(data)


def dataframe_for_lstm (df , n_step):
    df = deepcopy(df)
    df.set_index('Date',inplace=True)
    for i in range ( 1 , n_step+1 ):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)
    return df
shifted_df = dataframe_for_lstm(data,7)
print(shifted_df)

shifted_df.to_csv("AMZN_data_for_lstm.csv", index=False, encoding="utf-8-sig")