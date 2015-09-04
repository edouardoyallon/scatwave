--[[
     ScatWave implementation of Scattering Network
     Written by Edouard Oyallon
     Team DATA ENS
     Copyright 2015
]]

local tools={}

function tools.is_complex(x)
   return (x:size(x:dim())==2)
end

function tools.are_equal_dimension(x,y)
   return (torch.sum(torch.abs(torch.LongTensor(x:size())-torch.LongTensor(y:size())))==0)
end
function tools.are_equal(x,y)
   return (torch.sum(torch.abs(x-y))==0)
end

function tools.concatenateLongStorage(x,y)
      if(not x) then
         return y
      else
         local z=torch.LongStorage(#x+#y)
         for i=1,#x do
            z[i]=x[i]
         end   
         for i=1,#y do
            z[i+#x]=y[i]
         end
         return z
      end
   end


-- Very close from utils.lua in nn except that we need to copy the same stride..
function tools.recursiveType(param, type_str)
   if torch.type(param) == 'table' then
      for k, v in pairs(param) do
         param[k] = tools.recursiveType(v, type_str)
      end
   elseif torch.isTensor(param) then
      local st=param:stride()
      local si=param:size()
      local of=param:storageOffset()
      local stor=param:storage()
      local new_stor
      if(type_str=='torch.CudaTensor') then
         new_stor=torch.CudaStorage(stor:size())
      elseif (type_str=='torch.FloatTensor') then
         new_stor=torch.FloatStorage(stor:size())
      else
         error('Storage not supported')
      end
      new_stor:copy(stor)
      local tmp=torch.Tensor():type(type_str)
      param = tmp:set(new_stor,of,si,st)
   end
   return param
end

return tools