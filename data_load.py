import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd

data = pd.read_csv('./数据集KDD 99/data.csv')

X = data.drop(columns=['label'])
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50)


class LoadData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = torch.tensor(self.X.iloc[index])
        y = torch.tensor(self.y.iloc[index])
        return X, y


train_data = LoadData(X_train, y_train)
test_data = LoadData(X_test, y_test)

X_dimension = len(X_train.columns)
y_dimension = len(y_train.value_counts())
print(f"X的维度：{X_dimension}")
print(f"y的维度：{y_dimension}")
print(X_train.shape)
print(y_train.shape)

batch_size = 128

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
