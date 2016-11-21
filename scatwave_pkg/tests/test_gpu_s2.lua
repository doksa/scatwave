scatwave = require 'scatwave'
dofile('./env.lua')
require 'torch'
require 'cutorch'
torch.setdefaulttensortype('torch.FloatTensor')
dofile('./network_2d_translation_per.lua')
require 'sys'
local N=256
local J=5
local T=100
local x = torch.randn(1,1,N,N):cuda()
assert(x:isContiguous())
assert(x:size(2)==1)
--print('x size is',x:size())
local scat = scatwave.network_per.new(J,x:size())
scat:cuda()
sys.tic()
for t=1,T do
	local s2 = scat:scat(x)
	print(t)
end
print('s2 fwd per sample time is',sys.toc()/T)
