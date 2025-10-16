import torch
import torch.nn as nn

import pandas as pd
import numpy as np

from model import LSTM
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy

import matplotlib.pyplot as plt
from dataset import TimeSeriesDataset
from torch.utils.data import DataLoader
 



lookback = 7

data     = pd.read_csv('data/AMZN_data_for_lstm.csv')
dataAMZN = pd.read_csv('data/AMZN.csv')

dataAMZN['Date'] = pd.to_datetime(dataAMZN['Date'])
plt.plot(dataAMZN['Date'], dataAMZN['Close'])
plt.savefig("Recording/Original.jpg")
plt.close()

 


df_as_np = data.to_numpy()


scaler = MinMaxScaler(feature_range=(-1, 1))
df_as_np = scaler.fit_transform(df_as_np)
X = df_as_np[:,1:]
Y = df_as_np[:,0 ]
X = deepcopy(np.flip(X, axis=1))



n_splite = int(len(X)*0.95)
X_train = X[:n_splite]
X_test  = X[n_splite:]
Y_train = Y[:n_splite]
Y_test  = Y[n_splite:]
X_train = X_train.reshape((-1, lookback, 1))
X_test  = X_test.reshape((-1, lookback, 1))
Y_train = Y_train.reshape((-1, 1))
Y_test  = Y_test.reshape((-1, 1))


# Train :

X_train = torch.tensor(X_train).float()
Y_train = torch.tensor(Y_train).float()
X_test  = torch.tensor(X_test).float()
Y_test  = torch.tensor(Y_test).float()


train_dataset = TimeSeriesDataset(X_train, Y_train)
test_dataset  = TimeSeriesDataset(X_test , Y_test )
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

batch_size = 16
device = 'cuda'
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True )
test_loader  = DataLoader(test_dataset ,batch_size=batch_size,shuffle=False)


model = LSTM(1, 4, 1)
model.to(device)


num_epoch = 10
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters() , lr=0.001)





def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
    print()

def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('=================================')
    print()





for epoch in range(num_epoch):
    train_one_epoch()
    validate_one_epoch()



with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()




train_predictions = predicted.flatten()

# 逆標準化預測值
dummies = np.zeros((X_train.shape[0], lookback+1), dtype=np.float32)
dummies[:, 0] = train_predictions  # 已經是 numpy array
dummies = scaler.inverse_transform(dummies)
train_predictions = deepcopy(dummies[:, 0])

# 逆標準化實際值
dummies = np.zeros((X_train.shape[0], lookback+1), dtype=np.float32)
dummies[:, 0] = Y_train.detach().cpu().numpy().flatten()
dummies = scaler.inverse_transform(dummies)
new_y_train = deepcopy(dummies[:, 0])
 
plt.figure()
plt.plot(new_y_train, label='Actual Close')
plt.plot(train_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.savefig("Recording/TrainSet_Result.jpg")
plt.close()


# --- 測試集預測與逆標準化 ---
test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

# 逆標準化預測值
dummies = np.zeros((X_test.shape[0], lookback+1), dtype=np.float32)
dummies[:, 0] = test_predictions
dummies = scaler.inverse_transform(dummies)
test_predictions = deepcopy(dummies[:, 0])

# 逆標準化實際值
dummies = np.zeros((X_test.shape[0], lookback+1), dtype=np.float32)
dummies[:, 0] = Y_test.detach().cpu().numpy().flatten()
dummies = scaler.inverse_transform(dummies)
new_y_test = deepcopy(dummies[:, 0])

 
plt.figure()
plt.plot(new_y_test, label='Actual Close')
plt.plot(test_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.savefig("Recording/TestSet_Result.jpg")
plt.close()


 
torch.save(model.state_dict(), "Recording/lstm_model.pth")
print("Model saved at Recording/lstm_model.pth")
