local complex = {}
local tools= require 'tools'

function complex.unit_complex(alpha)
   local a=torch.Tensor({{torch.cos(alpha)},{torch.sin(alpha)}})
   a:resize(1,2)
   return a
end

function complex.modulus_wise(U)
   for k=1,#U do
      U[k].signal=complex.abs_value(U[k].signal)
   end
return U
end

function complex.abs_value(h)
   assert(tools.is_complex(h),'The number is not complex')
   local dim=h:size()
   local final_dim=torch.LongStorage(#dim-1)
   for i=1,h:nDimension()-1 do
      final_dim[i]=dim[i]   
   end
   local final_h=torch.pow(h,2)
   final_h=torch.sum(final_h,#dim)

   final_h:sqrt()

   return final_h:resize(final_dim)

end

function complex.realize(x)
   assert(tools.is_complex(x),'The number is not complex')
   return ((x:select(x:dim(),1)):clone())
end


function complex.multiply_complex_tensor(x,y)

   assert(tools.is_complex(x),'The number is not complex')
assert(tools.are_equal_dimension(x,y),'Dimensions of x and y differ')   

   xr=x:select(x:dim(),1)
   xi=x:select(x:dim(),2)
   yr=y:select(y:dim(),1)
   yi=y:select(y:dim(),2)

   
   --z=torch.Tensor(x:size()):fill(0)

   zr=torch.Tensor(x:size())

   zr:add(torch.cmul(xr,yr),torch.mul(torch.cmul(xi,yi),-1))

   zi=torch.Tensor(x:size())
   zi:add(torch.cmul(xr,yi),torch.cmul(xi,yr))
--   zi=torch.cmul(xr,yi)+torch.cmul(xi,yr)
--   gnuplot.imagesc(complex.abs_value(z))
   local z=torch.cat(zr,zi,x:dim())
return z
end

return complex