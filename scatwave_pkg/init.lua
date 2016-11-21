--[[
     ScatWave implementation of Scattering Network
     Written by Edouard Oyallon
     Team DATA ENS
     Copyright 2015
]]

require 'torch'

scatwave = {} -- define the global ScatNet table

scatwave.network = require 'scatwave.network_2d_translation'
scatwave.network_WT = require 'scatwave.network_2d_translation_WT'

require 'scatwave.op_fft2d'
--dofile('op_fft2d.lua')
require 'scatwave.op_ifft2d'
--dofile('op_ifft2d.lua')
require 'scatwave.op_down2d'
--dofile('op_down2d.lua')
require 'scatwave.op_mean2d'
--dofile('op_mean2d.lua')
require 'scatwave.op_lp2d'
require 'scatwave.op_modulus'
--dofile('op_lp2d.lua')
--dofile('op_convd2_infft.lua')
require 'scatwave.op_convd2_infft'
--dofile('op_convm2_infft.lua')
require 'scatwave.op_convm2_infft'

return scatwave
