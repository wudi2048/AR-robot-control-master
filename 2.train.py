import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from model import MLP
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset,DataLoader,random_split

class MyDataset(Dataset):
    def __init__(self,path):
        df = pd.read_csv(path,header=None)

        self.X = df.values[:,:-1]
        self.y = df.values[:,-1]

        self.X = self.X.astype('float32')
        self.y = LabelEncoder().fit_transform(self.y)


    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return [self.X[idx], self.y[idx]]

    def get_splits(self,n_test):

        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size

        return random_split(self,[train_size,test_size])

def main(args):

    print("Args:",args)

    dataset = MyDataset(args.path)
    train,test = dataset.get_splits(args.n_test)

    train_dataloader = DataLoader(train,batch_size=args.train_batchsize,shuffle=True)
    test_dataloader = DataLoader(test,batch_size=args.test_batchsize,shuffle=False)

    device = torch.device(args.device)
    model = MLP(args.n_inputs).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    losses = []
    for epoch in range(args.epochs):
        losses_iter = []
        for iter,(inputs,targets) in enumerate(train_dataloader):
            targets = targets.long()
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            y_hat = model(inputs)
            loss = criterion(y_hat,targets)
            loss.backward()
            optimizer.step()
            losses_iter.append(loss.cpu().data.item())
            print(f"epoch:{epoch}, batch:{iter}, loss:{loss.data}")
        losses.append(sum(losses_iter)/len(losses_iter))

    # Evaluate Model
    model.eval()
    predictions, actuals = [], []
    for iter, (inputs, targets) in enumerate(test_dataloader):
        inputs = inputs.to(device)
        y_hat = model(inputs).cpu()
        y_hat = y_hat.detach().numpy()  # just a 数据类型转换
        y_hat = np.argmax(y_hat,axis = 1)
        y_hat = y_hat.reshape((len(y_hat),1))


        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))

        predictions.append(y_hat)
        actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    acc = metrics.accuracy_score(actuals, predictions)
    print("Accuracy: %.3f" % acc)

    # Plot Loss
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.show()

    # Save Model
    save_dir = "save/"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model, save_dir + f"gestures1_model_e{args.epochs}.pth")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="./rawdata/gestures1_normal.txt")
    parser.add_argument('--n_test', type=float, default=0.3)
    # 123 = 175 * 0.7
    parser.add_argument('--train_batchsize', type=int, default=512)
    parser.add_argument('--test_batchsize', type=int, default=1024)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--n_inputs', type=int, default=18)
    parser.add_argument('--epochs', type=int, default=200)
    main(parser.parse_args())