--[[
     ScatWave implementation of Scattering Network
     Written by Edouard Oyallon
     Team DATA ENS
     Copyright 2015
]]

require 'torch'

local ScatNet = {}

ScatNet.network = require 'scatwave.network_2d_translation'
ScatNet.network_WT = require 'scatwave.network_2d_translation_WT'

return ScatNet
