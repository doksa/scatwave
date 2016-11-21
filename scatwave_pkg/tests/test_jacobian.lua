require 'torch'
require 'nn'
require 'scatwave'
torch.setdefaulttensortype('torch.FloatTensor')

mbdim = 3
dsj = 0
minv = -2
maxv = 2
pert = 1e-3
N = 4

--
a = torch.FloatTensor(1,1,N,N):fill(1)
m=scatwave.opDown2d(mbdim,dsj)
print('opDown2d testJacobian:',nn.Jacobian.testJacobian(m,a,minv,maxv,pert,verbose))

verbose = 1
a = torch.FloatTensor(1,1,N,N):fill(1)
m=scatwave.opFFT2d(mbdim)
print('opFFT2d testJacobian:',nn.Jacobian.testJacobian(m,a,minv,maxv,pert,verbose))
verbose = 0
--
a = torch.FloatTensor(1,1,N,N,2):fill(1) -- complex input
m=scatwave.opIFFT2d(mbdim,1) -- realout
print('opIFFT2d realout testJacobian:',nn.Jacobian.testJacobian(m,a,minv,maxv,pert,verbose))

a = torch.FloatTensor(1,1,N,N,2):fill(1) -- complex input
m=scatwave.opIFFT2d(mbdim,0) -- realout=0
print('opIFFT2d realout testJacobian:',nn.Jacobian.testJacobian(m,a,minv,maxv,pert,verbose))

filters_bank = require 'scatwave.filters_bank'
fft=require 'scatwave.wrapper_fft'
J=1
a = torch.FloatTensor(1,1,N,N):fill(1) -- real input
filters=filters_bank.modify(filters_bank.morlet_filters_bank_2D(a:size(),J,fft,0)) -- pad=0, no padding
ma=scatwave.opFFT2d(mbdim)
b = ma:forward(a)
m=scatwave.opConvDownsampleInFFT2d(filters.phi.signal[1],mbdim,dsj)
print('opConvDownsampleInFFT2d testJacobian:',nn.Jacobian.testJacobian(m,b,minv,maxv,pert,verbose))
--]]

a = torch.FloatTensor(1,1,N,N):fill(1) -- real input
a[1][1][1][1]=2 --torch.randn(a:size())
ma=scatwave.opFFT2d(mbdim)
a2 = ma:forward(a)
--print('input a2',a2)
m=scatwave.opModulus(mbdim)
print('opModulus testJacobian:',nn.Jacobian.testJacobian(m,a2,minv,maxv,pert,verbose))

--
m2=scatwave.opConvModulusInFFT2d(filters.psi[1].signal[1],mbdim,dsj)
print('opConvModulusInFFT2d testJacobian:',nn.Jacobian.testJacobian(m2,a2,minv,maxv,pert,verbose))
--]]
