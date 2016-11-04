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
            group:add(AbsBlock(nOut, nOut, false)):add(nn.Dropout(0.4))
      else
            group:add(AbsBlock(nOut, nOut, false))-- 90.4% :add(nn.Dropout(0.4))
      end
   end
   return group
end

local n = 5
absnet:add(nn.View(243,8,8))
absnet:add(backend.SpatialConvolution(243, 512, 3,3, 1,1, 1,1))
absnet:add(nn.SpatialBatchNormalization(512, 1e-3))--:add(nn.Dropout(0.4))
absnet:add(nn.ReLU())
absnet:add(AbsGroup(512,512,n,false))
absnet:add(AbsGroup(512,128,n,false))
absnet:add(backend.SpatialAveragePooling(8,8,1,1,0,0))
absnet:add(nn.Reshape(128))
absnet:add(nn.Linear(128,100))
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
