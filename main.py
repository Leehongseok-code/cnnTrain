import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch.optim as optim

torch.manual_seed(1)

#data loading
data=pd.read_csv('Data/csvdata/005930.Ks.csv')

#mid price computing
high=data['High'].values
low=data['Low'].values
mid=(high+low)/2

def minmax(data):
    numerator=data-min(data)
    denominator=max(data)-min(data)
    return numerator/(denominator+1e-7)


##차원 맞춰주기 input(187,7,1)  output(187,1,1)
def build_dataset(time_series,seq_length):#(train_data,7)
    datax=[]
    datay=[]
    for i in range(0,len(time_series)-seq_length):
        #print(time_series.shape)
        tempx=[]
        for j in range(i,i+seq_length):
            tempx.append([time_series[j]])
        _x=tempx
        #print(_x)
        #_x=time_series[i:i+seq_length,:]
        _y=time_series[[i+seq_length]]#다음 날 주가
        datax.append(_x)
        datay.append([_y])#억지로 차원 맞춰주기 위해서
    return np.array(datax),np.array(datay)

seq_length=7
input_dim=1
#output은 다음 날 주가 정보 하나이지만, hidden_dim=1이면 예측에 도움을 줄 수 있는 정보 사이즈 부족
hidden_dim=10
output_dim=1
learning_rate=0.01
iterations=5000
#input data 보정
xy=minmax(mid)
mid=mid[::-1]#reverse

train_size=int(len(xy)*0.8)
print("cnn train size:",train_size)
train_data=xy[0:train_size]
test_data=xy[train_size-seq_length:len(mid)]

trainx,trainy=build_dataset(train_data,seq_length)
testx,testy=build_dataset(test_data,seq_length)
#print(trainy)


trainx_tensor=torch.FloatTensor(trainx)
trainy_tensor=torch.FloatTensor(trainy)

testx_tensor=torch.FloatTensor(testx)
testy_tensor=torch.FloatTensor(testy)


class Cnn(nn.Module):
    def __init__(self,
                 kernel_size_1=1,#입력 차원
                 stride_size=1,
                 num_channels=7,
                 depth_1=1,
                 depth_2=5,
                 kernel_size_2=7,
                 num_hidden=20,
                 num_labels=7
                 ):
        super(Cnn, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv1d(num_channels, depth_1, kernel_size=kernel_size_1),

        )
        self.fc1 = nn.Sequential(
            nn.Linear(depth_2 * kernel_size_2, num_hidden),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(num_hidden, num_labels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.classifier(x)
        # x=self.fc1(x)
        # x=self.fc2(x)
        return (x)

#print(trainx_tensor)
model=Cnn()

#loss&optimizer setting
criterion=torch.nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(iterations):
    optimizer.zero_grad()
    #print(trainx_tensor.shape)
    outputs=model(trainx_tensor)
    #print(trainy_tensor.shape)
    cost=criterion(outputs,trainy_tensor)
    cost.backward()
    optimizer.step()
    print(epoch,cost.item())
    #print(trainx_tensor.shape,',',trainy_tensor.shape)

train_cnn=model(trainx_tensor).reshape(-1,1).data.numpy()
predict_cnn=model(testx_tensor).reshape(-1,1).data.numpy()


plt.plot(testy.reshape(-1))
plt.plot(model(testx_tensor).reshape(-1).data.numpy())
plt.legend(['original','prediction'])
plt.show()
