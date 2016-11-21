require 'torch'
require 'nn'

c = nn.MSECriterion(0)
t = torch.Tensor(1,10,1,4):fill(1)
s = torch.Tensor(1,10,1,4):fill(0)

print(c:forward(t,s))

