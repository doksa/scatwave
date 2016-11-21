scatwave = require 'scatwave'
dofile('./env.lua')
require 'torch'
require 'cutorch'
torch.setdefaulttensortype('torch.FloatTensor')
dofile('../network_2d_translation_per.lua')
require 'sys'
sys.tic()
local N=32
local J=3
--local x = torch.FloatTensor(1,1,N,N)
local x = torch.randn(1,1,N,N):cuda()
assert(x:isContiguous())
assert(x:size(2)==1)
--print('x size is',x:size())
local scat = scatwave.network_per.new(J,x:size())
scat:cuda()
local T=1
local sc = scat:scat_build()
sc:cuda()
for t=1,T do
	local s1 = sc:forward(x)
	local s2 = scat:scat(x)
	print('s1 size', s1:size())
	print('s2 size', s2:size())
	for k=1,s1:size(2) do
		--print('s1',k,s1[k]:size())
		local diff = s1:narrow(2,k,1):clone()
		--print(s2:narrow(3,k,1):size())
		print('diff',k,diff:add(-1,s2:narrow(3,k,1):resize(diff:size())):abs():max())
	end
	--print('s1',s1[3])
	--print('s2',s2:narrow(3,3,1))
end
print('fwd per sample time is',sys.toc()/T)
