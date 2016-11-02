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
      conv:add(nn.ReLU())      
   else
      conv:add(backend.SpatialConvolution(nIn, nOut, 3,3, 1,1, 1,1))
      conv:add(nn.SpatialBatchNormalization(nOut, 1e-3))
      conv:add(nn.ReLU())      
   end   
   return conv
end

-- a residual group contains n residual blocks
local function AbsGroup(nIn, nOut, n, firstSub)
   local group = nn.Sequential()
   group:add(AbsBlock(nIn, nOut, firstSub))
   for i = 1, n-1 do
      if(i%2==1) then
            group:add(AbsBlock(nOut, nOut, false)):add(nn.Dropout(0.6))
      else
            group:add(AbsBlock(nOut, nOut, false))-- 90.4% :add(nn.Dropout(0.4))
      end
   end
   return group
end

local n = 5

absnet:add(backend.SpatialConvolution(3,128,3,3,1,1,1,1))
absnet:add(nn.SpatialBatchNormalization(128, 1e-3))--:add(nn.Dropout(0.6))
   absnet:add(nn.ReLU())
<<<<<<< HEAD
=======
absnet:add(backend.SpatialConvolution(64,128,3,3,2,2,1,1))
absnet:add(nn.SpatialBatchNormalization(128, 1e-3))--:add(nn.Dropout(0.6))
   absnet:add(nn.ReLU())
>>>>>>> b3ac10f761f40046325b34318bb112a51d7c95e7
absnet:add(backend.SpatialConvolution(128,128,3,3,1,1,1,1))
absnet:add(nn.SpatialBatchNormalization(128, 1e-3))--:add(nn.Dropout(0.6))
   absnet:add(nn.ReLU())

absnet:add(backend.SpatialConvolution(128,256,3,3,1,1,1,1))
absnet:add(nn.SpatialBatchNormalization(256, 1e-3))--:add(nn.Dropout(0.6))
   absnet:add(nn.ReLU())

absnet:add(backend.SpatialConvolution(256,243,3,3,2,2,1,1))
absnet:add(nn.SpatialBatchNormalization(243, 1e-3))--:add(nn.Dropout(0.6))
absnet:add(nn.ReLU())

--absnet:add(nn.View(243,8,8))
--[[absnet:add(backend.SpatialConvolution(3,32,3,3,1,1,1,1))
absnet:add(nn.SpatialBatchNormalization(32, 1e-3))--:add(nn.Dropout(0.6))
   absnet:add(nn.ReLU())
absnet:add(backend.SpatialConvolution(32,64,3,3,2,2,1,1))
absnet:add(nn.SpatialBatchNormalization(64, 1e-3))--:add(nn.Dropout(0.6))
   absnet:add(nn.ReLU())
absnet:add(backend.SpatialConvolution(64,128,3,3,1,1,1,1))
absnet:add(nn.SpatialBatchNormalization(128, 1e-3))--:add(nn.Dropout(0.6))
   absnet:add(nn.ReLU())
absnet:add(backend.SpatialConvolution(128,256,3,3,2,2,1,1))
absnet:add(nn.SpatialBatchNormalization(256, 1e-3))--:add(nn.Dropout(0.6))
   absnet:add(nn.ReLU())
absnet:add(backend.SpatialConvolution(256,243,3,3,1,1,1,1))
absnet:add(nn.SpatialBatchNormalization(243, 1e-3))--:add(nn.Dropout(0.6))
absnet:add(nn.ReLU())88.5 !!!! ]]--
-- 512, 3,3, 1,1, 1,1))
--absnet:add(backend.SpatialConvolution(243, 512, 3,3, 1,1, 1,1))
--[[absnet:add(nn.SpatialBatchNormalization(512, 1e-3)):add(nn.Dropout(0.6))
absnet:add(nn.ReLU())
absnet:add(AbsGroup(512,512,n,false))
absnet:add(AbsGroup(512,128,n,false))
--absnet:add(AbsGroup(64,64,n,true))

absnet:add(backend.SpatialAveragePooling(8,8,1,1,0,0))
absnet:add(nn.Reshape(128))
absnet:add(nn.Linear(128,10))
absnet:add(nn.LogSoftMax())]]--

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
