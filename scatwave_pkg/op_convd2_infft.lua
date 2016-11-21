-- Author: Sixin Zhang (sixin.zhang@ens.fr)

local nn = require 'nn'
local complex = require 'scatwave.complex'

local opConvDownsampleInFFT2d, parent = torch.class('scatwave.opConvDownsampleInFFT2d', 'nn.Module')

-- input in fft (complex), output (real), filter in fft2 (complex) 
function opConvDownsampleInFFT2d:__init(filter_c, mbdim, dsj)
	parent.__init(self)
	self.filter_c = filter_c -- the filter
	self.mbdim = mbdim -- dim for the x
	self.dsj = dsj -- downsampling ratio = 2^dsj
	self.ifft = scatwave.opIFFT2d(self.mbdim,1) -- real output from ifft
	self.down = scatwave.opDown2d(self.mbdim,self.dsj)
	self.output_c = torch.Tensor()
end

function opConvDownsampleInFFT2d:updateOutput(input)
	if input:nElement() ~= self.output_c:nElement() then
		--print('opConvDownsampleInFFT2d forward alloc')
		self.output_c:resize(input:size()):fill(0)
	end
	-- output_c = input .* filter_c
	complex.multiply_complex_tensor_with_real_modified_tensor_in_place(input,self.filter_c,self.output_c)	
	-- output_r = ifft(output_c)
	local output_r = self.ifft:forward(self.output_c)
	-- output = downsample(output_r)
	self.output = self.down:forward(output_r)
	return self.output
end

function opConvDownsampleInFFT2d:updateGradInput(input, gradOutput)
	local gradDown = self.down:backward(self.ifft.output,gradOutput)
	local gradIfft = self.ifft:backward(self.output_c,gradDown)
	if gradIfft:nElement() ~= self.gradInput:nElement() then
		self.gradInput:resize(gradIfft:size())
	end
	complex.multiply_complex_tensor_with_real_modified_tensor_in_place(gradIfft,self.filter_c,self.gradInput)	
	return self.gradInput
end
