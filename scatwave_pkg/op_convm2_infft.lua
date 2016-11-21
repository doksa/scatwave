-- Author: Sixin Zhang (sixin.zhang@ens.fr)

local nn = require 'nn'
local complex = require 'scatwave.complex'
local tools = require 'scatwave.tools'

local opConvModulusInFFT2d, parent = torch.class('scatwave.opConvModulusInFFT2d', 'nn.Module')

-- complex input, real output, complex filter in fft2 
function opConvModulusInFFT2d:__init(filter_c, mbdim, dsj, eps)
	parent.__init(self)
	self.filter_c = filter_c -- the filter
	self.mbdim = mbdim -- dim for the x
	self.dsj = dsj -- downsampling ratio = 2^dsj
	self.ifft = scatwave.opIFFT2d(mbdim,0) -- complex output from ifft
	self.down = scatwave.opDown2d(mbdim,dsj)
	self.modulus = scatwave.opModulus(mbdim,eps)
	self.output_c = torch.Tensor()
end

function opConvModulusInFFT2d:updateOutput(input)
	if input:nElement() ~= self.output_c:nElement() then
		--print('opConvModulusInFFT2d forward alloc')
		assert(input:dim()==5)
		self.output_c:resize(input:size()):fill(0)
		local dsize = input:size()
		dsize[self.mbdim] = dsize[self.mbdim]/2^self.dsj
		dsize[self.mbdim+1] = dsize[self.mbdim+1]/2^self.dsj
		assert(dsize[5]==2)
		dsize[5] = 0
		self.output:resize(dsize)
	end
	-- output_c = input_c .* filter_c (this filter_c has same real and imag part, though filter is real in fft)
	complex.multiply_complex_tensor_with_real_modified_tensor_in_place(input,self.filter_c,self.output_c)
	-- output_cc = ifft(output_c) = input*filter
	-- TODO to reuse output_c for memory efficient, be careful with backprop as well
	local output_cc = self.ifft:forward(self.output_c)
	local output_c2 = self.down:forward(output_cc)
	self.output = self.modulus:forward(output_c2)
	return self.output
end

function opConvModulusInFFT2d:updateGradInput(input, gradOutput)
	local gradModulus = self.modulus:backward(self.down.output,gradOutput)
	local gradDown = self.down:backward(self.ifft.output,gradModulus)
	local gradIfft = self.ifft:backward(self.output_c,gradDown)
	if gradIfft:nElement() ~= self.gradInput:nElement() then
		--print('opConvModulusInFFT2d backward alloc')
		self.gradInput:resize(gradIfft:size())
	end
	complex.multiply_complex_tensor_with_real_modified_tensor_in_place(gradIfft,self.filter_c,self.gradInput)
	return self.gradInput
end
