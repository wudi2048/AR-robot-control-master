
import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self,n_inputs):
        super(MLP, self).__init__()

        self.hidden1 = nn.Linear(n_inputs,256)
        nn.init.kaiming_uniform_(self.hidden1.weight,nonlinearity='relu')
        self.act1 = nn.ReLU()

        self.hidden2 = nn.Linear(256,256)
        nn.init.kaiming_uniform_(self.hidden2.weight,nonlinearity='relu')
        self.act2 = nn.ReLU()

        self.hidden3 = nn.Linear(256, 256)
        nn.init.kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = nn.ReLU()

        self.hidden4 = nn.Linear(256, 256)
        nn.init.kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act4 = nn.ReLU()

        self.hidden5 = nn.Linear(256, 256)
        nn.init.kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        self.act5 = nn.ReLU()

        self.hidden6 = nn.Linear(256, 256)
        nn.init.kaiming_uniform_(self.hidden6.weight, nonlinearity='relu')
        self.act6 = nn.ReLU()

        self.hidden7 = nn.Linear(256, 256)
        nn.init.kaiming_uniform_(self.hidden7.weight, nonlinearity='relu')
        self.act7 = nn.ReLU()

        self.hidden8 = nn.Linear(256,5)
        nn.init.xavier_uniform_(self.hidden8.weight)
        self.act8 = nn.Softmax(dim=1)

    def forward(self,X):
        X = self.hidden1(X)
        X = self.act1(X)

        X = self.hidden2(X)
        X = self.act2(X)

        X = self.hidden3(X)
        X = self.act3(X)

        X = self.hidden4(X)
        X = self.act4(X)

        X = self.hidden5(X)
        X = self.act5(X)

        X = self.hidden6(X)
        X = self.act6(X)

        X = self.hidden7(X)
        X = self.act7(X)

        X = self.hidden8(X)
        X = self.act8(X)

        return X