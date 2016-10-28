require 'xlua'
require 'optim'
require 'cunn'
require 'cudnn'
require 'mattorch'

scatwave = require 'scatwave'


local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs2")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default damned_absolute_value)     model name
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

print('selecting W1')
--local model = nn.Sequential()
--local model_aug_data = nn.Sequential()
--model_aug_data:add(nn.BatchFlip():float())
--model_aug_data:add(nn.RandomCrop(4):float())
--model_aug_data:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())

--model:add(dofile('models/'..opt.model..'.lua'):cuda())
model=torch.load('logs_whitened/model.net')
out=model:get(1):get(2).weight
mattorch.save('operator_W1.mat',out:double())



