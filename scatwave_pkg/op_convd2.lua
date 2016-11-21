-- Author: Sixin Zhang (sixin.zhang@ens.fr)

local nn = require 'nn'

local scatwave = require 'scatwave'
local complex = require 'scatwave.complex'
local filters_bank = require 'scatwave.filters_bank'
local conv_lib = require 'scatwave.conv_lib'
local tools = require 'scatwave.tools'

local opConvDownsample2d, parent = torch.class('scatwave.opConvDownsample2d', 'nn.Module')

-- real input, real output, real filter in fft2 
function opConvDownsample2d:__init(filter_c, mbdim, dsj)
	parent.__init(self)
	self.filter_c = filter_c -- the filter
	self.mbdim = mbdim -- dim for the x
	self.dsj = dsj -- downsampling ratio = 2^dsj
	self.input_c = nil
	self.output_c = torch.Tensor()
	self.output_r = torch.Tensor()
end

function opConvDownsample2d:type(_type, tensorCache)
	if(_type=='torch.CudaTensor') then
		self.fft=require 'scatwave.cuda/wrapper_CUDA_fft_nvidia'
	elseif(_type=='torch.FloatTensor') then
		self.fft=require 'scatwave.wrapper_fft'
	else
		error('This type is not handled by scatwave')
	end
	parent.type(self, _type, tensorCache)
end

function opConvDownsample2d:updateOutput(input)
	if input:nElement() ~= self.output_r:nElement() then
		--print('alloc')
		assert(input:dim()==4)
		self.output_r:resize(input:size()):fill(0)
		self.output_c:resize(tools.concatenateLongStorage(input:size(),torch.LongStorage({2}))):fill(0)
		self.input_c = self.output_c -- :resize(tools.concatenateLongStorage(input:size(),torch.LongStorage({2}))):fill(0)
		local dsize = input:size()
		dsize[self.mbdim] = dsize[self.mbdim]/2^self.dsj
		dsize[self.mbdim+1] = dsize[self.mbdim+1]/2^self.dsj
		self.output:resize(dsize):fill(0)
	end
	-- input_c = fft(input)
	self.fft.my_2D_fft_real_batch(input,self.mbdim,self.input_c)
	-- output_c = input_c .* filter_c
	complex.multiply_complex_tensor_with_real_modified_tensor_in_place(self.input_c,self.filter_c,self.output_c)
	-- output_r = ifft(output_c)
	self.fft.my_2D_ifft_complex_to_real_batch(self.output_c,self.mbdim,self.output_r)
	self.output:copy(conv_lib.downsample_2D_inplace(
						 self.output_r,self.dsj,self.mbdim))
	return self.output
end

function opConvDownsample2d:updateGradInput(input, gradOutput)
	assert(false)
	return self.gradInput
end
