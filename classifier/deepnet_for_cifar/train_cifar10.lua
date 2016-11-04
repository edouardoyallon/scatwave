require 'xlua'
require 'optim'
require 'cunn'
require 'image'

scatwave = require 'scatwave'
local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs_cifar10")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 30)          epoch step
   --model                    (default generic_model_cifar10)     model name
   --max_epoch                (default 300)           maximum number of iterations
]]



-----------------------------------------------
---- SCATTERING
-----------------------------------------------

J=2
args={}
SCAT=scatwave.network.new(2,torch.LongStorage({128,3,32,32}))
SCAT:cuda()


SCAT_test=scatwave.network.new(2,torch.LongStorage({125,3,32,32}))
SCAT_test:cuda()


-----------------------------------------------
----
-----------------------------------------------


print(opt)


do -- random crop
   local RandomCrop, parent = torch.class('nn.RandomCrop', 'nn.Module')

   function RandomCrop:__init(pad)
      assert(pad)
      parent.__init(self)
      self.pad = pad
      self.module = nn.SpatialZeroPadding(pad,pad,pad,pad)
      self.train = true
   end

   function RandomCrop:updateOutput(input)
      assert(input:dim())
      if self.train then
         local padded = self.module:forward(input)
         local x = torch.random(1,self.pad*2 + 1)
         local y = torch.random(1,self.pad*2 + 1)
         self.output = padded:narrow(4,x,32):narrow(3,y,32)
      else
         self.output:set(input)
      end
      return self.output
   end

   function RandomCrop:type(type)
      self.module:type(type)
      return parent.type(self, type)
   end
end



do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output = input
    return self.output
  end
end

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
local model_aug_data = nn.Sequential()
model_aug_data:add(nn.BatchFlip():float())
model_aug_data:add(nn.RandomCrop(4):float())
model_aug_data:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())

model:add(dofile('models/'..opt.model..'.lua'):cuda())
print(model)

print(c.blue '==>' ..' loading data')
provider = torch.load 'cifar10_whitened.t7'
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()

confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
criterion = nn.CrossEntropyCriterion():cuda()


print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}







-----------------------------------------------
---- SCATTERING
-----------------------------------------------   

inputs_cuda = torch.CudaTensor(128,3,32,32)
outputs_cuda = torch.CudaTensor(128,3,81,8,8)

inputs_cuda_test = torch.CudaTensor(125,3,32,32)
outputs_cuda_test = torch.CudaTensor(125,3,81,8,8)


-----------------------------------------------
----
-----------------------------------------------






function train()
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = torch.CudaTensor(opt.batchSize)
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = provider.trainData.data:index(1,v)
    targets:copy(provider.trainData.labels:index(1,v))

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
-----------------------------------------------
---- SCATTERING
-----------------------------------------------   
            inputs_cuda:copy(model_aug_data:forward(inputs))
            outputs_cuda:copy(SCAT:scat(inputs_cuda,1))

-----------------------------------------------
----
-----------------------------------------------

       local outputs = model:forward(outputs_cuda)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(outputs_cuda, df_do)

      confusion:batchAdd(outputs, targets)

      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1
end


function test()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 125
  for i=1,provider.testData.data:size(1),bs do


-----------------------------------------------
---- SCATTERING
-----------------------------------------------  
              inputs=provider.testData.data:narrow(1,i,bs)      
 inputs_cuda_test:copy(inputs)
      outputs_cuda_test:copy(SCAT_test:scat(inputs_cuda_test,1))
-----------------------------------------------
----
-----------------------------------------------

    local outputs = model:forward(outputs_cuda_test)


    confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))


  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion.totalValid * 100}
    testLogger:style{'-','-'}
    testLogger:plot()

    local base64im
    do
      os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
      os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
      local f = io.open(opt.save..'/test.base64')
      if f then base64im = f:read'*all' end
    end

    local file = io.open(opt.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  -- save model every 50 epochs
  if epoch % 60 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model)
  end

  confusion:zero()
end


for i=1,opt.max_epoch do
  train()
  test()
end


