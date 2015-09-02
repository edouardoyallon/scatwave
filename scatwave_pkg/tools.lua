local tools={}

function tools.is_complex(x)
   return (x:size(x:dim())==2)
end

function tools.are_equal_dimension(x,y)
   return (torch.sum(torch.abs(torch.LongTensor(x:size())-torch.LongTensor(y:size())))==0)
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

return tools