local scatwave = require 'scatwave'

require 'sys'
require 'cutorch'

sys.tic()

local N=256
local J=5
x=torch.CudaTensor(1,1,N,N)
scat = scatwave.network.new(J,x:size())
scat:cuda()

local T=100
for t=1,T do
	sx = scat:scat(x)
	print(t)
end
print('fwd per sample time is',sys.toc()/T)
