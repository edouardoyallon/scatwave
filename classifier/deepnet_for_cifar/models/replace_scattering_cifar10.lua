require('nn')
require('cunn')
require('cudnn')


local backend = cudnn
local absnet = nn.Sequential()


absnet:add(backend.SpatialConvolution(3,128,3,3,1,1,1,1))
absnet:add(nn.SpatialBatchNormalization(128, 1e-3))
absnet:add(nn.ReLU())
absnet:add(backend.SpatialConvolution(128,128,3,3,2,2,1,1))
absnet:add(nn.SpatialBatchNormalization(128, 1e-3))
absnet:add(nn.ReLU())
absnet:add(backend.SpatialConvolution(128,128,3,3,1,1,1,1))
absnet:add(nn.SpatialBatchNormalization(128, 1e-3))
absnet:add(nn.ReLU())
absnet:add(backend.SpatialConvolution(128,256,3,3,1,1,1,1))
absnet:add(nn.SpatialBatchNormalization(256, 1e-3))
absnet:add(nn.ReLU())
absnet:add(backend.SpatialConvolution(256,243,3,3,2,2,1,1))
absnet:add(nn.SpatialBatchNormalization(243, 1e-3))
absnet:add(nn.ReLU())

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
