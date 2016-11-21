-- test lena's scat transform
require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'image'
local img=image.load('lena.ppm',3,'byte')
img = img:float():div(255):mean(1)
--print(img:size(),img:max(),img:min())

--require 'cutorch'
local N=img:size(3)
local mb = torch.FloatTensor(1,1,N,N):copy(img)
--print(mb:max(),mb:min())

dofile('./network_2d_translation_per.lua')
local J=3
local scat = scatwave.network_per.new(J,mb:size())

--[[
require 'mattorch'
--mattorch.save('scatwave_filter_phi_1.mat',{phi1=scat.filters.phi.signal[1]:double()})
for j=1,J*8 do
    mattorch.save('scatwave_filter_psi_' .. j .. '.mat',
                  {psi=scat.filters.psi[j].signal[1]:double()})
end
--]]

--[[
local sc = scat:scat_build()
sc:float()
local s1 = sc:forward(mb)
print(s1:size())
require 'mattorch'
mattorch.save('test_lena.mat',{sx=s1:double()})
--]]

local sc = scat:scat_lp()
sc:float()
local s1 = sc:forward(mb) 
print(s1:size())

--require 'mattorch'
--mattorch.save('test_lena_scatlp_order1.mat',{scatlp=s1:double()})
--]]
