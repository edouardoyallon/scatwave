require('nn')
require('cunn')
require('cudnn')


local backend = cudnn
local absnet = nn.Sequential()

-- a block
local function AbsBlock(nIn, nOut, subsample)
   local conv = nn.Sequential()
   if subsample then      
      conv:add(backend.SpatialConvolution(nIn, nOut, 3,3, 2,2, 1,1))
      conv:add(nn.SpatialBatchNormalization(nOut, 1e-3))
      conv:add(nn.Abs())      
   else
      conv:add(backend.SpatialConvolution(nIn, nOut, 3,3, 1,1, 1,1))
      conv:add(nn.SpatialBatchNormalization(nOut, 1e-3))
      conv:add(nn.Abs())      
   end   
   return conv
end

-- a residual group contains n residual blocks
local function AbsGroup(nIn, nOut, n, firstSub)
   local group = nn.Sequential()
   group:add(AbsBlock(nIn, nOut, firstSub))
   for i = 1, n-1 do
      group:add(AbsBlock(nOut, nOut, false))
   end
   return group
end

local n = 2
absnet:add(nn.View(243,8,8))
absnet:add(backend.SpatialConvolution(243, 128, 3,3, 1,1, 1,1))
absnet:add(nn.SpatialBatchNormalization(128, 1e-3))
absnet:add(AbsGroup(128,64,n,false))
absnet:add(AbsGroup(64,64,n,false))
--absnet:add(AbsGroup(64,64,n,true))
absnet:add(nn.Abs())
absnet:add(backend.SpatialAveragePooling(8,8,1,1,0,0))
absnet:add(nn.Reshape(64))
absnet:add(nn.Linear(64,10))
absnet:add(nn.LogSoftMax())

-- initialization from MSR
local function MSRinit(net)
   local function init(name)
      for k,v in pairs(net:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         v.bias:zero()
      end
   end
   -- have to do for both backends
   init'cudnn.SpatialConvolution'
   init'nn.SpatialConvolution'
end

MSRinit(absnet)

return absnet
