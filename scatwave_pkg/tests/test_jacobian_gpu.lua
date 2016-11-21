require 'torch'
require 'nn'
require 'scatwave'
local tools = require 'scatwave.tools'

mbdim = 3
dsj = 0
minv = -2
maxv = 2
pert = 1e-3
N = 4
J = 1

torch.setdefaulttensortype('torch.FloatTensor')
a = torch.FloatTensor(1,1,N,N)
filters_bank = require 'scatwave.filters_bank'
fft=require 'scatwave.wrapper_fft' -- cuda/wrapper_CUDA_fft_nvidia'
filters=filters_bank.modify(filters_bank.morlet_filters_bank_2D(a:size(),J,fft,0))

require 'cutorch'
require 'cunn'

torch.setdefaulttensortype('torch.CudaTensor')

--
a = torch.CudaTensor(1,1,N,N):fill(1)
m=scatwave.opDown2d(mbdim,dsj)
print('opDown2d testJacobian:',nn.Jacobian.testJacobian(m,a,minv,maxv,pert,verbose))
--]]

--verbose = 1
a = torch.CudaTensor(1,1,N,N):fill(1)
m=scatwave.opFFT2d(mbdim)
m:cuda()
print('opFFT2d testJacobian:',nn.Jacobian.testJacobian(m,a,minv,maxv,pert,verbose))

--
a = torch.CudaTensor(1,1,N,N,2):fill(1) -- complex input
m=scatwave.opIFFT2d(mbdim,1)
m:cuda() -- realout
print('opIFFT2d realout testJacobian:',nn.Jacobian.testJacobian(m,a,minv,maxv,pert,verbose))

a = torch.CudaTensor(1,1,N,N,2):fill(1) -- complex input
m=scatwave.opIFFT2d(mbdim,0)
m:cuda() -- realout=0
print('opIFFT2d realout testJacobian:',nn.Jacobian.testJacobian(m,a,minv,maxv,pert,verbose))

a = torch.CudaTensor(1,1,N,N):fill(1) -- real input
-- pad=0, no padding
tools.recursiveType(filters,'torch.CudaTensor')
ma=scatwave.opFFT2d(mbdim)
ma:cuda()
b = ma:forward(a)
m=scatwave.opConvDownsampleInFFT2d(filters.phi.signal[1],mbdim,dsj)
m:cuda()
print('opConvDownsampleInFFT2d testJacobian:',nn.Jacobian.testJacobian(m,b,minv,maxv,pert,verbose))
--

a = torch.CudaTensor(1,1,N,N):fill(1) -- real input
a[1][1][1][1]=2 --torch.randn(a:size())
ma=scatwave.opFFT2d(mbdim)
ma:cuda()
a2 = ma:forward(a)
--print('input a2',a2)
m=scatwave.opModulus(mbdim)
m:cuda()
print('opModulus testJacobian:',nn.Jacobian.testJacobian(m,a2,minv,maxv,pert,verbose))

--
m2=scatwave.opConvModulusInFFT2d(filters.psi[1].signal[1],mbdim,dsj)
m2:cuda()
print('opConvModulusInFFT2d testJacobian:',nn.Jacobian.testJacobian(m2,a2,minv,maxv,pert,verbose))
--]]
