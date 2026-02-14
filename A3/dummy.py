import random
import numpy as np
import torch as tr

seed = 0
random.seed(seed)
np.random.seed(seed)
tr.manual_seed(seed)

class SimpleFeedForwardNet(tr.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = tr.nn.Linear(784, 512, bias=True)
        self.linear2 = tr.nn.Linear(512, 256, bias=True)
        self.linear3 = tr.nn.Linear(256, 128, bias=True)
        self.linear4 = tr.nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x

model = SimpleFeedForwardNet()
optimizer = tr.optim.Adam(model.parameters(), lr=0.001)
loss_fn = tr.nn.CrossEntropyLoss()
