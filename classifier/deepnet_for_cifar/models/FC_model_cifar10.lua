require('nn')
require('cunn')
require('cudnn')


local backend = cudnn
local absnet = nn.Sequential()

absnet:add(nn.View(243*8*8))
absnet:add(nn.Linear(8*8*243,2048))
absnet:add(nn.ReLU())
absnet:add(nn.Linear(2048,2048))
absnet:add(nn.BatchNormalization(2048,1e-3))
absnet:add(nn.ReLU())
absnet:add(nn.Linear(2048,10))
absnet:add(nn.LogSoftMax())

-- initialization from MSR
local function MSRinit(net)
   local function init(name)
      for k,v in pairs(net:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(8/n))
         v.bias:zero()
      end
   end
   -- have to do for both backends
   init'cudnn.SpatialConvolution'
   init'nn.SpatialConvolution'
end

MSRinit(absnet)

return absnet
