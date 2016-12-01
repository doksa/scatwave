-- Author: Sixin Zhang (sixin.zhang@ens.fr)

local nn = require 'nn'
local conv_lib = require 'scatwave.conv_lib'

local opLP2d, parent = torch.class('scatwave.opLP2d', 'nn.Module')

-- input is real
function opLP2d:__init(lplist,mbdim)
	assert(mbdim == 3 and #lplist > 0) -- assume minibatch case, otherwise modify the condition in updateOutput alloc
	parent.__init(self)
	self.lplist = lplist
	self.lp = nn.Sequential()
	local ccat = nn.ConcatTable()
	for i,v in pairs(self.lplist) do
		if v==-1 then
			local lp_sum = nn.Sequential()
			lp_sum:add(scatwave.opMean2d(mbdim+1,mbdim+1))
			lp_sum:add(scatwave.opMean2d(mbdim,mbdim+1))
			ccat:add(lp_sum)
		elseif v==2 then
			local lp_sq = nn.Sequential()
			lp_sq:add(nn.Square())
			lp_sq:add(scatwave.opMean2d(mbdim+1,mbdim+1))
			lp_sq:add(scatwave.opMean2d(mbdim,mbdim+1))
			lp_sq:add(nn.Sqrt())
			ccat:add(lp_sq)
		end
	end
	self.lp:add(ccat)
	self.lp:add(nn.JoinTable(mbdim+1,mbdim+1))
	self.mbdim = mbdim -- dim for the x
end

function opLP2d:updateOutput(input)
	--print('in opLP2d forward',input:size())
	self.output = self.lp:forward(input)
	--print(self.output:size())
	return self.output
end

function opLP2d:updateGradInput(input, gradOutput)
	self.gradInput = self.lp:updateGradInput(input, gradOutput)
	return self.gradInput
end
