--[[
	ScatWave implementation of Scattering Network
	Written by Edouard Oyallon
	Team DATA ENS
	Copyright 2015
]]

local complex = require 'scatwave.complex'
local filters_bank = require 'scatwave.filters_bank'
local conv_lib = require 'scatwave.conv_lib'
local tools = require 'scatwave.tools'

local scatwave = require 'scatwave'
local network = torch.class('network_per')
scatwave.network_per = network

function network:__init(J,U0_dim)
	if(J==nil) then
		error('Please specify the scale.')
	else
		self.J=J
	end
	
	if(U0_dim==nil) then
		error('Please specify the size.')
	else
		self.U0_dim=U0_dim
	end
	
	self.fft=require 'scatwave.wrapper_fft'         
	self.filters=filters_bank.modify(filters_bank.morlet_filters_bank_2D(U0_dim,J,self.fft,0)) -- pad=0, no padding
	
	-- Allocate temporary variables
	local size_multi_res=self.filters.size_multi_res   
	--print_r(size_multi_res)
	--print_r(self.filters.stride_multi_res)
	--assert(false)
	function allocate_multi_res(isComplex)
		local x={}      
		for r=1,#size_multi_res do         
			local s
			if(isComplex) then
				s = tools.concatenateLongStorage(size_multi_res[r],torch.LongStorage({2}))
			else
				s=size_multi_res[r]
			end
			x[r]=torch.FloatTensor(s):fill(0)
		end
		return x
	end
	-- Input signal
	local size_r_2 = torch.LongStorage(#U0_dim):copy(U0_dim)
	size_r_2[U0_dim:size()-1]=size_multi_res[1][U0_dim:size()-1]
	self.U0_r_1 = torch.FloatTensor(U0_dim):fill(0) -- Auxiliary variable for padding because Torch is not efficient enough
	self.U0_r_2  = torch.FloatTensor(size_r_2):fill(0)
	self.U0_c = torch.FloatTensor(tools.concatenateLongStorage(size_multi_res[1],torch.LongStorage({2}))):fill(0)
	
	-- First layer temporary variables
	self.U1_c = allocate_multi_res(1)
	self.U1_r = allocate_multi_res()
	
	-- Second layer temporary variables
	self.U2_c = allocate_multi_res(1)
	self.U2_r = allocate_multi_res()
	
	-- Allocate the averaged layer
	local size_S = torch.LongStorage(#U0_dim+1)
	for l=1,#U0_dim-2 do
		size_S[l]=U0_dim[l]
	end
	size_S[#U0_dim-1]=1 + J*8 + J*(J-1)/2*8*8 -- number of layers
	size_S[#size_S-1] = 1+torch.floor((U0_dim[#U0_dim]-1)/2^J)
	size_S[#size_S] = 1+torch.floor((U0_dim[#U0_dim]-1)/2^J)
	self.S_r = torch.FloatTensor(size_S):fill(0)
end

-- This function is pretty much copied from Module.lua in nn @ https://github.com/torch/nn/blob/master/Module.lua
function network:type(_type)   
	if(_type=='torch.CudaTensor') then
		self.fft=require 'scatwave.cuda/wrapper_CUDA_fft_nvidia'
	elseif(_type=='torch.FloatTensor') then
		self.fft=require 'scatwave.wrapper_fft'
	else
		error('This type is not handled by scatwave')
	end
	
	for key,param in pairs(self) do
		self[key] = tools.recursiveType(param,_type)
	end
	
	return self
end

function network:float()
	self:type('torch.FloatTensor')
end

function network:cuda()
	self:type('torch.CudaTensor')
end

function network:get_filters()
	return self.filters
end

function network:scat_lp(lplist)
	local lplist = lplist or {-1,2} -- l2 norm and , -1 means computing the mean
	local filters = self.filters
	local mbdim = #self.U0_dim-1
	local join_dim = mbdim-1 -- join at the color channel
	local ndim = #self.U0_dim
	local J = self.J

	-- 0th order
	local opH_J = scatwave.opLP2d(lplist,mbdim)
	-- 1st order
	local sc1a = nn.ConcatTable()
	sc1a:add(opH_J)
	local seq1a = nn.Sequential()
	seq1a:add(scatwave.opFFT2d(mbdim))
	local sc1b = nn.ConcatTable()
	for j1=1,#filters.psi do -- j1 and j2 index both scale and angel (J*8 in total)
		local seq1b = nn.Sequential()
		local J1=filters.psi[j1].j
		local opG_j1 = scatwave.opConvModulusInFFT2d(filters.psi[j1].signal[1],mbdim,J1)
		local opH_j1_J = scatwave.opLP2d(lplist,mbdim)
		seq1b:add(opG_j1)
		local sc2a = nn.ConcatTable()
		sc2a:add(opH_j1_J)
		--2nd order
		local seq2a = nn.Sequential()
		seq2a:add(scatwave.opFFT2d(mbdim))
		local sc2b = nn.ConcatTable()
		for j2=1,#filters.psi do
			local J2=filters.psi[j2].j         
			if (J2>J1) then
				local seq2b = nn.Sequential()
				local opG_j2 = scatwave.opConvModulusInFFT2d(filters.psi[j2].signal[J1+1],mbdim,(J2-J1))
				local opH_j2_J = scatwave.opLP2d(lplist,mbdim)
				seq2b:add(opG_j2)
				seq2b:add(opH_j2_J)
				sc2b:add(seq2b)
			end
		end
		-- check if to add this branch
		if #(sc2b.modules) > 0 then
			seq2a:add(sc2b)
			seq2a:add(nn.JoinTable(join_dim,ndim))
			sc2a:add(seq2a)
		end
		--]]
		seq1b:add(sc2a)
		seq1b:add(nn.JoinTable(join_dim,ndim))
		sc1b:add(seq1b) -- 1st order scattering
	end
	seq1a:add(sc1b)
	seq1a:add(nn.JoinTable(join_dim,ndim))
	sc1a:add(seq1a)
	local sc = nn.Sequential()
	sc:add(sc1a)
	sc:add(nn.JoinTable(join_dim,ndim))
	return sc
end

function network:scat_build()
	--local fft = self.fft -- Routines used for the computations  
	local filters = self.filters
	local mini_batch_ndim = #self.U0_dim-1
	--print('mini_batch_ndim',mini_batch_ndim)
	local J = self.J
	--print(J)
	-- build the scat network, 1st order
	local sc1 = nn.ConcatTable()
	local opH_J = scatwave.opConvDownsampleInFFT2d(filters.phi.signal[1],mini_batch_ndim,J)
	sc1:add(opH_J)
	--print('#psi=',#filters.psi)
	for j1=1,#filters.psi do
		--print('j1='..j1)
		local J1=filters.psi[j1].j
		local seq_j1 = nn.Sequential()
		local opG_j1 = scatwave.opConvModulusInFFT2d(filters.psi[j1].signal[1],mini_batch_ndim,J1)
		local opH_j1_J = scatwave.opConvDownsampleInFFT2d(filters.phi.signal[J1+1],mini_batch_ndim,(J-J1))
		seq_j1:add(opG_j1)
		--2nd order
		local sc2 = nn.ConcatTable()
		sc2:add(opH_j1_J)
		for j2=1,#filters.psi do
			local J2=filters.psi[j2].j         
			if (J2>J1) then
				local seq_j2 = nn.Sequential()
				local opG_j2 = scatwave.opConvModulusInFFT2d(filters.psi[j2].signal[J1+1],mini_batch_ndim,(J2-J1))
				local opH_j2_J = scatwave.opConvDownsampleInFFT2d(filters.phi.signal[J2+1],mini_batch_ndim,(J-J2))
				seq_j2:add(opG_j2)
				seq_j2:add(scatwave.opFFT2d(mini_batch_ndim))
				seq_j2:add(opH_j2_J)
				sc2:add(seq_j2)
			end
		end
		seq_j1:add(scatwave.opFFT2d(mini_batch_ndim))
		seq_j1:add(sc2)
		seq_j1:add(nn.JoinTable(mini_batch_ndim-1,#self.U0_dim))
		--]]
		sc1:add(seq_j1)
	end
	--print('sc size',sc1:size())
	local sc = nn.Sequential()
	sc:add(scatwave.opFFT2d(mini_batch_ndim))
	sc:add(sc1)
	sc:add(nn.JoinTable(mini_batch_ndim-1,#self.U0_dim))
	return sc
end

function network:scat_build0()
	--local fft = self.fft -- Routines used for the computations  
	local filters = self.filters
	local mini_batch_ndim = #self.U0_dim-1
	--print('mini_batch_ndim',mini_batch_ndim)
	local J = self.J
	--print(J)
	-- build the scat network, 1st order
	local sc1 = nn.ConcatTable()
	local opH_J = scatwave.opConvDownsample2d(filters.phi.signal[1],mini_batch_ndim,J)
	sc1:add(opH_J)
	--print('#psi=',#filters.psi)
	for j1=1,#filters.psi do
		--print('j1='..j1)
		local J1=filters.psi[j1].j
		local seq_j1 = nn.Sequential()
		local opG_j1 = scatwave.opConvModulus2d(filters.psi[j1].signal[1],mini_batch_ndim,J1)
		local opH_j1_J = scatwave.opConvDownsample2d(filters.phi.signal[J1+1],mini_batch_ndim,(J-J1))
		seq_j1:add(opG_j1)
		--2nd order
		local sc2 = nn.ConcatTable()
		sc2:add(opH_j1_J)
		for j2=1,#filters.psi do
			local J2=filters.psi[j2].j         
			if (J2>J1) then
				local seq_j2 = nn.Sequential()
				local opG_j2 = scatwave.opConvModulus2d(filters.psi[j2].signal[J1+1],mini_batch_ndim,(J2-J1))
				local opH_j2_J = scatwave.opConvDownsample2d(filters.phi.signal[J2+1],mini_batch_ndim,(J-J2))
				seq_j2:add(opG_j2)
				seq_j2:add(opH_j2_J)
				sc2:add(seq_j2)
			end
		end
		seq_j1:add(sc2)
		seq_j1:add(nn.JoinTable(mini_batch_ndim-1,#self.U0_dim))
		--]]
		sc1:add(seq_j1)
	end
	--print('sc size',sc1:size())
	local sc = nn.Sequential()
	sc:add(sc1)
	sc:add(nn.JoinTable(mini_batch_ndim-1,#self.U0_dim))
	return sc
end

-- Here, we minimize the creation of memory to avoid using garbage collector
function network:scat(U0_r,doPeriodize)
	--   assert(U0_r:isSize(self.U0_dim),'Not the correct specified input size') -- Does not exist with cuda tensor..
	assert(U0_r:isContiguous(),'Input tensor is not contiguous')
	assert(doPeriodize == nil)
	-- Routines used for the computations
	local fft = self.fft  
	local filters = self.filters
	
	-- Minibatch variables (size, number of dim)
	local mini_batch_ndim = #self.U0_dim-1
	
	-- Input signal
	local U0_r_1 = self.U0_r_1 -- Auxiliary variable for padding because Torch is not efficient enough
	local U0_r_2 = self.U0_r_2     
	local U0_c = self.U0_c
	
	-- First layer temporary variables
	local U1_c = self.U1_c
	local U1_r = self.U1_r
	
	-- Second layer temporary variables
	local U2_c = self.U2_c
	local U2_r = self.U2_r
	
	-- Averaged scattering final variable
	local count_S_r = 1   
	local S_r = self.S_r
	local J = self.J
	
	-- Pad the signal along the first, then second dimension. This operation can not be done jointly because Torch does not handle tensor copy in readonly mode
	--conv_lib.pad_signal_along_k(U0_r, filters.size_multi_res[1][#filters.size_multi_res[1]-1], mini_batch_ndim,U0_r_1)
	--conv_lib.pad_signal_along_k(U0_r_1, filters.size_multi_res[1][#filters.size_multi_res[1]-1], mini_batch_ndim+1,U0_r_2)      
	U0_r_2:copy(U0_r)
	--print(U0_r_1:size(),U0_r_2:size())
	--assert(false)
	
	-- All the computations will be done with the padded signal. Only one unpadding will be applied on the output averaged signal.
	local function unpad_output_signal(ds)
		-- print(ds:size(),S_r:size())
		return  ds -- :narrow(mini_batch_ndim,2,S_r:size(mini_batch_ndim+1)):narrow(mini_batch_ndim+1,2,S_r:size(mini_batch_ndim+2)) 
	end
	
	-- FFT of the input image
	fft.my_2D_fft_real_batch(U0_r_2,mini_batch_ndim,U0_c)    

	
	-- Compute the multiplication with xf and the LF, store it in U1_c[1]
	complex.multiply_complex_tensor_with_real_modified_tensor_in_place(U0_c,filters.phi.signal[1],U1_c[1])
	
	if(doPeriodize) then   
		-- Compute the complex to real iFFT of U1_c[1] and store it in U1_r[1]
		complex.periodize_in_place(U1_c[1],J,mini_batch_ndim,U1_c[J+1])
		-- Compute the iFFT of U1_c[J+1], and store it in U1_r[J+1]      
		fft.my_2D_ifft_complex_to_real_batch(U1_c[J+1],mini_batch_ndim,U1_r[J+1])
		S_r:narrow(mini_batch_ndim,count_S_r,1):copy(unpad_output_signal(U1_r[J+1]))
	else
		-- Compute the complex to real iFFT of U1_c[1] and store it in U1_r[1]
		fft.my_2D_ifft_complex_to_real_batch(U1_c[1],mini_batch_ndim,U1_r[1])
		-- Store the downsample in S[k] where k is the corresponding position in the memory, k<-k+1   
		S_r:narrow(mini_batch_ndim,count_S_r,1):copy(unpad_output_signal(conv_lib.downsample_2D_inplace(U1_r[1],J,mini_batch_ndim)))
	end
	count_S_r=count_S_r+1
	
	for j1=1,#filters.psi do
		-- Compute the multiplication with xf and the filters which is real in Fourier, finally store it in U1_c[1]
		local J1=filters.psi[j1].j
		complex.multiply_complex_tensor_with_real_modified_tensor_in_place(U0_c,filters.psi[j1].signal[1],U1_c[1])
		
		
		
		if(doPeriodize) then
			if(J1>0) then
				complex.periodize_in_place(U1_c[1],J1,mini_batch_ndim,U1_c[J1+1]) 
			end
			
			fft.my_2D_fft_complex_batch(U1_c[J1+1],mini_batch_ndim,1,U1_c[J1+1])
			
		else
			-- Compute the iFFT of U1_c[1], and store it in U1_c[1]      
			fft.my_2D_fft_complex_batch(U1_c[1],mini_batch_ndim,1,U1_c[1])
			-- We subsample it manually by changing its stride and store the subsampling in U1_c[j1]      
			U1_c[J1+1]:copy(conv_lib.downsample_2D_inplace(U1_c[1],J1,mini_batch_ndim))
		end
		
		
		-- Compute the modulus and store it in U1_r[j1]
		complex.abs_value_inplace(U1_c[J1+1],U1_r[J1+1])
		
		-- Compute the Fourier transform and store it in U1_c[j1]
		fft.my_2D_fft_real_batch(U1_r[J1+1],mini_batch_ndim,U1_c[J1+1])

		
		
		-- Compute the multiplication with U1_c[j1] and the LF, store it in U2_c[j1]
		complex.multiply_complex_tensor_with_real_modified_tensor_in_place(U1_c[J1+1],filters.phi.signal[J1+1],U2_c[J1+1])
		
		if(doPeriodize) then            
			complex.periodize_in_place(U2_c[J1+1],J-J1,mini_batch_ndim,U1_c[J+1])
			
			fft.my_2D_ifft_complex_to_real_batch(U1_c[J+1],mini_batch_ndim,U1_r[J+1])
			S_r:narrow(mini_batch_ndim,count_S_r,1):copy(unpad_output_signal(U1_r[J+1]))
		else
			-- Compute the iFFT complex to real of U2_c[j1] and store it in U1_r[j1]
			fft.my_2D_ifft_complex_to_real_batch(U2_c[J1+1],mini_batch_ndim,U1_r[J1+1])
			-- Store the downsample in S_r
			S_r:narrow(mini_batch_ndim,count_S_r,1):copy(unpad_output_signal(conv_lib.downsample_2D_inplace(U1_r[J1+1],J-J1,mini_batch_ndim)))
			--print('store s at loc',count_S_r)
		end
		count_S_r=count_S_r+1
		--
		for j2=1,#filters.psi do
			local J2=filters.psi[j2].j         
			if (J2>J1) then
				-- Compute the multiplication with U1_c[j1] and the filters, and store it in U2_c[j1]
				--print('filter psi',j2,J1+1,'size',filters.psi[j2].signal[J1+1]:size(),U1_c[J1+1]:size(),U2_c[J1+1]:size())
				complex.multiply_complex_tensor_with_real_modified_tensor_in_place(U1_c[J1+1],filters.psi[j2].signal[J1+1],U2_c[J1+1])
				
				if(doPeriodize) then     
					complex.periodize_in_place(U2_c[J1+1],J2-J1,mini_batch_ndim,U2_c[J2+1])
					fft.my_2D_fft_complex_batch(U2_c[J2+1],mini_batch_ndim,1,U2_c[J2+1])
				else
					-- Compute the iFFT of U2_c[j1], and store it in U2_c[j1]
					fft.my_2D_fft_complex_batch(U2_c[J1+1],mini_batch_ndim,1,U2_c[J1+1])         
					
					-- Subsample it and store it in U2_c[j2]
					U2_c[J2+1]:copy(conv_lib.downsample_2D_inplace(U2_c[J1+1],J2-J1,mini_batch_ndim))
				end
				--print('U2_c[J2+1] size is',U2_c[J2+1]:size())
				
				-- Compute the modulus and store it in U2_r[j2]
				complex.abs_value_inplace(U2_c[J2+1],U2_r[J2+1])
				
				
				-- Compute the Fourier transform of U2_r[j2] and store it in U2_c[j2]
				fft.my_2D_fft_real_batch(U2_r[J2+1],mini_batch_ndim,U2_c[J2+1])
				
				-- Compute the multiplication with U2_c[j2] and the LF, store it in U2_c[j2]    
				complex.multiply_complex_tensor_with_real_modified_tensor_in_place(U2_c[J2+1],filters.phi.signal[J2+1],U2_c[J2+1])
				
				
				if(doPeriodize) then
					complex.periodize_in_place(U2_c[J2+1],J-J2,mini_batch_ndim,U2_c[J+1])
					fft.my_2D_ifft_complex_to_real_batch(U2_c[J+1],mini_batch_ndim,U2_r[J+1])
					S_r:narrow(mini_batch_ndim,count_S_r,1):copy(unpad_output_signal(U2_r[J+1]))
				else
					-- Compute the complex to real iFFT of U2_c[j2] and store it in U2_r[j2]
					fft.my_2D_ifft_complex_to_real_batch(U2_c[J2+1],mini_batch_ndim,U2_r[J2+1])
					
					-- Store the downsample in S_r
					S_r:narrow(mini_batch_ndim,count_S_r,1):copy(unpad_output_signal(conv_lib.downsample_2D_inplace(U2_r[J2+1],J-J2,mini_batch_ndim)))
				end            
				count_S_r=count_S_r+1 				
			end
		end
		--]]
	end
	
	return S_r
end

return network
