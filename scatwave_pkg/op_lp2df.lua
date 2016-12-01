-- Author: Sixin Zhang (sixin.zhang@ens.fr)

local nn = require 'nn'
local conv_lib = require 'scatwave.conv_lib'

local opLP2df, parent = torch.class('scatwave.opLP2df', 'nn.Module')

-- input is real
function opLP2df:__init(lplist,mbdim)
	assert(mbdim == 3 and #lplist > 0) -- assume minibatch case, otherwise modify the index
	parent.__init(self)
	self.lplist = lplist
	self.eps = eps or 1e-16
	for i,v in pairs(self.lplist) do
		if v==2 then
			self.input_v2=torch.Tensor()
		end
	end
	self.mbdim = mbdim -- dim for the x
end

function opLP2df:updateOutput(input)
	if input:size(1)*input:size(2)*#self.lplist ~= self.output:nElement() then
		--print('in opLP2df forward alloc',input:size())
		self.output:resize(input:size(1),input:size(2),1,#self.lplist)
		if self.input_v2 then
			self.input_v2:resize(input:size())
		end
	end
	if self.input_v2 then
		self.input_v2:copy(input)
		self.input_v2:cmul(input)
	end
	for ip,v in pairs(self.lplist) do
		for i1=1,input:size(1) do
			for i2=1,input:size(2) do
				local inp=input:select(1,i1):select(1,i2)
				local out=self.output:select(1,i1):select(1,i2):select(1,1)
				if v==-1 then
					out[ip] = inp:mean()
				elseif v==2 then
					local inp2=self.input_v2:select(1,i1):select(1,i2)
					out[ip] = math.sqrt(inp2:mean())
				end
			end
		end
	end
	return self.output
end

function opLP2df:updateGradInput(input, gradOutput)
	if self.gradInput:nElement() ~= input:nElement() then
		--print('in opLP2df backward alloc',input:size())
		self.gradInput:resize(input:size()):fill(0)
	end
	self.gradInput:zero()
	for ip,v in pairs(self.lplist) do
		for i1=1,input:size(1) do
			for i2=1,input:size(2) do
				local gin=self.gradInput:select(1,i1):select(1,i2)
				local out=self.output:select(1,i1):select(1,i2):select(1,1)
				local gout=gradOutput:select(1,i1):select(1,i2):select(1,1)
				if v==-1 then
					gin:add(gout[ip]/gin:nElement())
				elseif v==2 then
					gin:add(gout[ip]/((out[ip]+self.eps)*gin:nElement()),input)
				end
			end
		end
	end
	return self.gradInput
end
