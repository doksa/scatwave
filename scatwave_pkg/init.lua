--[[
     ScatWave implementation of Scattering Network
     Written by Edouard Oyallon
     Team DATA ENS
     Copyright 2015
]]

require 'torch'

scatwave = {} -- define the global ScatNet table

scatwave.network = require 'scatwave.network_2d_translation'
scatwave.network_per = require 'scatwave.network_2d_translation_per'

require 'scatwave.op_fft2d'
require 'scatwave.op_ifft2d'
require 'scatwave.op_down2d'
require 'scatwave.op_mean2d'
require 'scatwave.op_lp2d'
require 'scatwave.op_lp2df'
require 'scatwave.op_modulus'
require 'scatwave.op_convd2_infft'
require 'scatwave.op_convm2_infft'

return scatwave
