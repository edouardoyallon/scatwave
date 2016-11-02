require 'xlua'
require 'optim'
require 'cutorch'
require 'cunn'
require 'cudnn'

require 'mattorch'

scatwave=require 'scatwave'
local c = require 'trepl.colorize'


J=2
args={}
SCAT=scatwave.network.new(2,torch.LongStorage({128,3,32,32}))
SCAT:cuda()


SCAT_test=scatwave.network.new(2,torch.LongStorage({125,3,32,32}))
SCAT_test:cuda()


opt={}

opt = lapp[[
   -s,--save                  (default "logs_retrain_cifar10")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 1)        learning rate (must be set to the same as before)
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 30)          epoch step
   --model_pretrain                    (default logs_cifar10/model.net)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --loggy                    (default nil)           for batch mode saving log files
]]

print(opt)

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
--model = nn.Sequential()
--model:add(nn.Sequential():add(dofile(opt.model_train..'.lua'):cuda()))

model = torch.load(opt.model_pretrain):cuda()
W1 = mattorch.load('operator_sparsified_W1.mat')
W1=W1.W1_p:cuda()
model_fix=nn.Sequential()
model_fix:add(model:get(1):get(1):clone())
model_fix:add(model:get(1):get(2):clone())
model_fix:add(model:get(1):get(3):clone())
model_fix:add(model:get(1):get(4):clone())
model_fix:add(model:get(1):get(5):clone())
--model_fix=cudnn.SpatialConvolution(3,3,...)
print(W1:size())
print(model_fix:get(2).weight:size())

model_fix:get(2).weight:copy(W1)

for i=1,5 do
   model:get(1):remove(1)
end



   model_fix:evaluate()

--model:add(nn.BatchFlip():float())
--model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda()) --<- EDOUARD
--model:add(nn.Sequential():add(dofile('models/'..opt.model..'.lua'):cuda())) <-- EDOUARD

--model:get(2).updateGradInput = function(input) return end
print(model_fix)
print(model)

print(c.blue '==>' ..' loading data')
provider = torch.load 'cifar10_whitened.t7'

confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
print('lol')
parameters,gradParameters = model:getParameters()
print('lol')

print(c.blue'==>' ..' setting criterion')
criterion = nn.CrossEntropyCriterion():cuda()


print(c.blue'==>' ..' configuring optimizer')
optimState = {
   learningRate = opt.learningRate,
   weightDecay = opt.weightDecay,
   momentum = opt.momentum,
   learningRateDecay = opt.learningRateDecay,
}
optimMethod = optim.sgd


inputs_cuda = torch.CudaTensor(128,3,32,32)
outputs_cuda = torch.CudaTensor(128,3,81,8,8)
outputs_cuda_fix = torch.CudaTensor(128,512,8,8)

inputs_cuda_test = torch.CudaTensor(125,3,32,32)
outputs_cuda_test = torch.CudaTensor(125,3,81,8,8)
outputs_cuda_fix_test = torch.CudaTensor(125,512,8,8)

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
            ------ SCATTERING IS APPLIED HERE      
            inputs_cuda:copy(inputs)
            outputs_cuda:copy(SCAT:scat(inputs_cuda,1))            
            outputs_cuda_fix:copy(model_fix:forward(outputs_cuda))
            ------------------------------
            local outputs = model:forward(outputs_cuda_fix)
            local f = criterion:forward(outputs, targets)
            local df_do = criterion:backward(outputs, targets)
            model:backward(outputs_cuda_fix, df_do)
            
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


function test(the_end)
   -- disable flips, dropouts and batch normalization
   model:evaluate()
   print(c.blue '==>'.." testing")
   local n = provider.testData.data:size(1)
   local bs = 125
   for i=1,n,bs do
      inputs=provider.testData.data:narrow(1,i,bs)      
         ------ SCATTERING IS APPLIED HERE      
         inputs_cuda_test:copy(inputs)
      outputs_cuda_test:copy(SCAT_test:scat(inputs_cuda_test,1))

      outputs_cuda_fix_test:copy(model_fix:forward(outputs_cuda_test))
      ------------------------------
      _,y = model:forward(outputs_cuda_fix_test):max(2)
      confusion:batchAdd(y,provider.testData.labels:narrow(1,i,bs))
   end
   
   confusion:updateValids()
   print('Test accuracy:', confusion.totalValid * 100)
   
   if testLogger then
      paths.mkdir(opt.save)
      testLogger:add{train_acc, confusion.totalValid * 100}
      testLogger:style{'-','-'}
      -- testLogger:plot()
      local file = io.open(opt.save..'/report2.html','w')
      local header = [[
    <!DOCTYPE html>
    <html>
    <body>
    <title>$SAVE - $EPOCH</title>
    <img src="test.png">
    ]]
      header = header:gsub('$SAVE',opt.save):gsub('$EPOCH',epoch)
      file:write(header)
      file:write'<h4>optimState:</h4>\n'
      file:write'<table>\n'
      for k,v in pairs(optimState) do
         if torch.type(v) == 'number' then
            file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
         end
      end
      file:write'</table><pre>\n'
      file:write(tostring(confusion)..'\n')
      file:write(tostring(model)..'\n')
      file:write("</pre></body></html>")
      file:close()
      os.execute('convert -density 200 '..opt.save..'/test.log.eps '..opt.save..'/test.png')
   end
   
   -- save model every 25 epochs
   if epoch % 25 == 0 then
      local filename = paths.concat(opt.save, 'model_half.net')
      print('==> saving model to '..filename)
      torch.save(filename, model)
   end
   if(not the_end) then
      confusion:zero()
   end
end


function test2(the_end)
   -- disable flips, dropouts and batch normalization
   model:evaluate()
   print(c.blue '==>'.." testing")
   local n = provider.testData.data:size(1)
   local bs = 125
   for i=1,n,bs do
      inputs=provider.testData.data:narrow(1,i,bs)      
         ------ SCATTERING IS APPLIED HERE      
         inputs_cuda_test:copy(inputs)
      outputs_cuda_test:copy(SCAT_test:scat(inputs_cuda_test,1))

      outputs_cuda_fix_test:copy(model_fix:forward(outputs_cuda_test))
      ------------------------------
      _,y = model:forward(outputs_cuda_fix_test):max(2)
      confusion:batchAdd(y,provider.testData.labels:narrow(1,i,bs))
   end
   
   confusion:updateValids()
   print('Test accuracy:', confusion.totalValid * 100)
  
   if(not the_end) then
      confusion:zero()
   end
end

print('We test the model before fine tuning')
test2()
for i=1,opt.max_epoch do
  
   
train()
    if(i<opt.max_epoch) then
      test()
   else
      test(1)
   end
end

if(opt.loggy) then
   local  file = io.open(opt.loggy,'a+')
   file:write("here:\n")
   file:write(tostring(model)..'\n')   
   file:write(tostring(confusion)..'\n')
   file:write('Test accuracy:'..(confusion.totalValid*100)..'\n\n')
   file:close()
end


