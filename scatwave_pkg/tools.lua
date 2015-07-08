local tools={}

function tools.is_complex(x)
   return (x:size(x:dim())==2)
end

function tools.are_equal_dimension(x,y)
   return (torch.sum(torch.abs(torch.LongTensor(x:size())-torch.LongTensor(y:size())))==0)
end


function tools.start_with_equal_dimension(x,y)
local a=0   
   for i=1,torch.min(torch.Tensor({{x:nDimension(),y:nDimension()}}))-1 do
      a=a+torch.abs(x:size(i)-y:size(i))
   end
return a
end
   --return (torch.sum(torch.abs(torch.LongTensor(x:size())-torch.LongTensor(y:size())))==0)
--end 

return tools