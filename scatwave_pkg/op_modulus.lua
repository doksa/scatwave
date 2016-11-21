-- Author: Sixin Zhang (sixin.zhang@ens.fr)

local nn = require 'nn'
local complex = require 'scatwave.complex'
local tools = require 'scatwave.tools'

local opModulus, parent = torch.class('scatwave.opModulus', 'nn.Module')

-- input in complex
function opModulus:__init(mbdim,eps)
	parent.__init(self)
	self.mbdim = mbdim
	self.eps = eps or 1e-8
	self.input_c = torch.Tensor()
	self.output_eps = torch.Tensor()
end

function opModulus:updateOutput(input)
	if input:nElement() ~= self.output:nElement()*2 then
		--print('opModulus forward alloc')
		local dsize = input:size()
		self.input_c:resize(dsize)
		dsize[self.mbdim] = dsize[self.mbdim]
		dsize[self.mbdim+1] = dsize[self.mbdim+1]
		assert(dsize[5]==2)
		dsize[5] = 0
		self.output:resize(dsize):fill(0)
	end
	self.input_c:copy(input)
	complex.abs_value_inplace(self.input_c,self.output) -- this destorys self.input_c
	return self.output
end

function opModulus:updateGradInput(input, gradOutput)
	if self.gradInput:nElement() ~= gradOutput:nElement()*2 then
		--print('opModulus backward alloc')
		self.gradInput:resize(
			tools.concatenateLongStorage(gradOutput:size(),torch.LongStorage({2}))):fill(0)
		self.output_eps:resize(self.output:size()):fill(0)
	end
	self.output_eps:copy(self.output)
	self.output_eps:add(self.eps) -- TODO use self.output instead for memory efficiency?
	complex.divide_real_and_complex_tensor(input, self.output_eps, self.gradInput) -- compute phase
	-- compute gradModulus = phase (complex) .* gradOutput (real)
	complex.multiply_complex_tensor_with_real_tensor_in_place(self.gradInput,gradOutput,self.gradInput)
	return self.gradInput
end
