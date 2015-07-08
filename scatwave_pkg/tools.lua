local tools={}

function tools.is_complex(x)
   return (x:size(x:dim())==2)
end

function tools.are_equal_dimension(x,y)
   return (torch.sum(torch.abs(torch.LongTensor(x:size())-torch.LongTensor(y:size())))==0)
end
  

return tools