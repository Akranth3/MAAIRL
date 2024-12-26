import torch
from networks import ActorNetwork

model = ActorNetwork(8, 1, 64, 64) 
print(model(torch.randn(1, 8)))