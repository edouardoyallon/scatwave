local complex = {}
require 'torch'
local tools= require 'tools'


function complex.unit_complex(alpha)
   assert(torch.isTensor(alpha),'You can only input a tensor to unit_complex')
   return torch.cat(torch.cos(alpha),torch.sin(alpha),alpha:nDimension()+1)
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
   return final_h:view(final_dim)   
end

function complex.realize(x)
   assert(tools.is_complex(x),'The number is not complex')

   return ((x:select(x:dim(),1)):clone())
end

function complex.realize_inplace(x)
   assert(tools.is_complex(x),'The number is not complex')
   x:narrow(x:nDimension(),2,1):fill(0)
   return x

end


function complex.multiply_complex_tensor(x,y,mini_batch_x,myTensor)
   assert(tools.is_complex(x),'The number is not complex')
   
   local strides=torch.LongStorage(x:nDimension())
   for l=1,x:nDimension() do
      if(l<=mini_batch_x) then
         strides[l]=0
      else
         strides[l]=x:stride(l)
      end
   end
   y_=myTensor(y:storage(),y:storageOffset(),x:size(),strides) -- hint, set the stride to 0 when you wanna minibatch...
   
   assert(tools.are_equal_dimension(x,y_),'Dimensions of x and y differ')      
   local xr=x:select(x:dim(),1)
   local xi=x:select(x:dim(),2)
   local yr=y_:select(y_:dim(),1)
   local yi=y_:select(y_:dim(),2)
   
   local z=myTensor(x:size()):fill(0)   
   local z_real = z:select(z:dim(), 1)
   local z_imag = z:select(z:dim(), 2)
      
   
   torch.cmul(z_real, xr, yr)
   z_real:addcmul(-1, xi, yi)   
   torch.cmul(z_imag, xr, yi)
   z_imag:addcmul(1, xi, yr)      

   return z
end


function complex.multiply_real_and_complex_tensor(x,y,myTensor)
   
   assert(tools.is_complex(x),'First input must be complex')
   assert(not tools.is_complex(y),'Second input must be real')
      
   local xr=x:select(x:dim(),1)
   local xi=x:select(x:dim(),2)
   local yr=y
   
   local z=myTensor(x:size()):fill(0)
   local z_real = z:select(z:dim(), 1)
   local z_imag = z:select(z:dim(), 2)
   
   torch.cmul(z_real, xr, yr)
   torch.cmul(z_imag, xi, yr)
   return z
end

function complex.multiply_complex_tensor_with_real_tensor_in_place(x,y,output)
   assert(tools.is_complex(x),'First input should be complex')
   
   output:narrow(output:nDimension(),1,1):cmul(x:narrow(x:nDimension(),1,1),y)
   output:narrow(output:nDimension(),2,1):cmul(x:narrow(x:nDimension(),1,1),y)
   
end


function complex.multiply_complex_number_and_real_tensor(x,y,myTensor)
   
   assert(tools.is_complex(x),'First input must be complex')
   assert(not tools.is_complex(y),'Second input must be real')
      
   local xr=x[1]
   local xi=x[2]
   
   local fs=torch.LongStorage(y:nDimension()+1)
   for l=1,y:nDimension() do
      fs[l]=y:size(l)
   end
   fs[y:nDimension()+1]=2
   
   local z=myTensor(fs):fill(0)
   local z_real = z:select(z:dim(), 1)
   local z_imag = z:select(z:dim(), 2)

   z_real:add(xr,y) 
      z_imag:add(xi,y)

   return z
end

return complex