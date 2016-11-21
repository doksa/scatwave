-- Author: Sixin Zhang (sixin.zhang@ens.fr)

local nn = require 'nn'

local scatwave = require 'scatwave'
local complex = require 'scatwave.complex'
local filters_bank = require 'scatwave.filters_bank'
local conv_lib = require 'scatwave.conv_lib'
local tools = require 'scatwave.tools'

local opConvModulus2d, parent = torch.class('scatwave.opConvModulus2d', 'nn.Module')

-- real input, real output, complex filter in fft2 
function opConvModulus2d:__init(filter_c, mbdim, dsj)
	parent.__init(self)
	self.filter_c = filter_c -- the filter
	self.mbdim = mbdim -- dim for the x
	self.dsj = dsj -- downsampling ratio = 2^dsj
	self.input_c = nil
	self.output_c = torch.Tensor()
	self.output_c2 = torch.Tensor()
end

function opConvModulus2d:type(_type, tensorCache)
	if(_type=='torch.CudaTensor') then
		self.fft=require 'scatwave.cuda/wrapper_CUDA_fft_nvidia'
	elseif(_type=='torch.FloatTensor') then
		self.fft=require 'scatwave.wrapper_fft'
	else
		error('This type is not handled by scatwave')
	end
	parent.type(self, _type, tensorCache)
end

function opConvModulus2d:updateOutput(input)
	if input:nElement()*2 ~= self.output_c:nElement() then
		--print('alloc')
		assert(input:dim()==4)
		self.output_c:resize(tools.concatenateLongStorage(input:size(),torch.LongStorage({2}))):fill(0)
		self.input_c = self.output_c
		local dsize = input:size()
		dsize[self.mbdim] = dsize[self.mbdim]/2^self.dsj
		dsize[self.mbdim+1] = dsize[self.mbdim+1]/2^self.dsj
		self.output_c2:resize(tools.concatenateLongStorage(dsize,torch.LongStorage({2}))):fill(0)
		self.output:resize(dsize):fill(0)
	end
	-- input_c = fft(input)
	self.fft.my_2D_fft_real_batch(input,self.mbdim,self.input_c)
	-- output_c = input_c .* filter_c
	complex.multiply_complex_tensor_with_real_modified_tensor_in_place(self.input_c,self.filter_c,self.output_c)
	-- output_c = ifft(output_c) = input*filter
	self.fft.my_2D_fft_complex_batch(self.output_c,self.mbdim,1,self.output_c)
	-- output_c2 = downsample(output_c)
	self.output_c2:copy(conv_lib.downsample_2D_inplace(
							self.output_c,self.dsj,self.mbdim))
	-- output = abs(output_c2)
	complex.abs_value_inplace(self.output_c2,self.output)
	return self.output
end

function opConvModulus2d:updateGradInput(input, gradOutput)
	assert(false)
	return self.gradInput
end
