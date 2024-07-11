import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# lets create a data
# import lib for data from sklearn
from sklearn.datasets import make_swiss_roll

# create data
def create_sroll(n_samples=1000, noise=0.0, random_state=None):
    data, _ = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=random_state)
    data = data[:, [0, 2]]
    data = data.astype(np.float32)
    data = torch.from_numpy(data)
    return data

# define a model (linear)
class encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(encoder, self).__init__()
        self.mlp_point = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.max_pool = nn.max_pool1d()
        self.mlp_feature = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.mlp_point(x)
        x = self.max_pool(x)
        x = self.mlp_feature(x)
        return x

if __name__ == '__main__':
    # create data
    data = create_sroll()
    # plot data 
    plt.scatter(data[:, 0], data[:, 1], c='r', s=1)
    plt.show()
    # create model
    model = encoder(2, 100, 2)
    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # create loss
    loss = nn.MSELoss()
    # train model
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(data)
        l = loss(output, data)
        l.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('epoch: {}, loss: {}'.format(epoch, l.item()))
    # plot data
    plt.scatter(data[:, 0], data[:, 1], c='r')
    plt.scatter(output[:, 0], output[:, 1], c='b')
    plt.show()

    