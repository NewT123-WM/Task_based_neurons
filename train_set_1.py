import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset
from utils.nolinear_neuron import Neurons
from utils.seeds import random_seed
from utils.dataset import MyData
import h5py


def main(group, sr, seed):
    test_size = 0.2
    batch_size = 10
    max_epoch = 300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with h5py.File('./data/epsilon_low.h5', 'r') as f:
        X_all = np.array(f[group]['x_train'])
        y_all = np.array(f[group]['y_train'])

    scaler = MinMaxScaler()
    X_all = scaler.fit_transform(X_all)
    y_all = scaler.fit_transform(y_all)

    X_all = torch.from_numpy(X_all).to(torch.float32)
    y_all = torch.from_numpy(y_all).to(torch.float32)
    X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=10)
    trainset = MyData(X, y)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    random_seed(seed)
    net1 = nn.Sequential(nn.Linear(10, 2),
                         nn.ReLU(),

                         nn.Linear(2, 1),
                         ).to(device)

    cost1 = nn.MSELoss().to(device)
    optimizer1 = torch.optim.RMSprop(net1.parameters(), lr=0.01)

    for k in range(max_epoch):
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            predict = net1(inputs)
            loss1 = cost1(predict, labels)
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

    # print("net1 training done -------------")

    with torch.no_grad():
        train_error_net1 = cost1(net1(X.to(device)), y.to(device))

    with torch.no_grad():
        test_error_net1 = cost1(net1(X_test.to(device)), y_test.to(device))

    # print("net1 testing  done -------------\n")

    random_seed(seed)
    net2 = nn.Sequential(Neurons(10, 1, sr)
                         ).to(device)

    cost2 = nn.MSELoss().to(device)
    optimizer2 = torch.optim.RMSprop(net2.parameters(), lr=0.001)

    for k in range(max_epoch):
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            predict = net2(inputs)
            loss2 = cost2(predict, labels)
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

    with torch.no_grad():
        train_error_net2 = cost2(net2(X.to(device)), y.to(device))

    with torch.no_grad():
        test_error_net2 = cost2(net2(X_test.to(device)), y_test.to(device))

    return train_error_net1.cpu(), test_error_net1.cpu(), train_error_net2.cpu(), \
        test_error_net2.cpu()


if __name__ == '__main__':
    gr = 'no1'
    expr = '6@x**2 + 2@x - 3'

    result = np.empty((1, 8))

    train_mse_net1_list = []
    test_mse_net1_list = []
    train_mse_net2_list = []
    test_mse_net2_list = []

    for j in range(10):
        train_net1, test_net1, train_net2, test_net2 = main(group=gr, sr=expr, seed=(j + 1) * 10)
        train_mse_net1_list.append(train_net1)
        test_mse_net1_list.append(test_net1)
        train_mse_net2_list.append(train_net2)
        test_mse_net2_list.append(test_net2)

    result[1, 0] = np.mean(train_mse_net1_list)
    result[1, 1] = np.mean(test_mse_net1_list)
    result[1, 2] = np.mean(train_mse_net2_list)
    result[1, 3] = np.mean(test_mse_net2_list)
    result[1, 4] = np.std(train_mse_net1_list)
    result[1, 5] = np.std(test_mse_net1_list)
    result[1, 6] = np.std(train_mse_net2_list)
    result[1, 7] = np.std(test_mse_net2_list)

    print('====================测试结束=======================')
    np.savetxt(gr + '_result.csv', result, delimiter=',')
