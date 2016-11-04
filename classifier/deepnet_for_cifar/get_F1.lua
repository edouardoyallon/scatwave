require 'xlua'
require 'optim'
require 'cunn'
require 'cudnn'
require 'mattorch'

scatwave = require 'scatwave'


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
model=torch.load('logs_cifar10/model.net')
out=model:get(1):get(2).weight
mattorch.save('operator_W1.mat',out:double())



