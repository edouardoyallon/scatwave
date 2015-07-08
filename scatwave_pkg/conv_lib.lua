local conv_lib={}
local tools=require 'tools'
local fftw_complex_cast = 'fftw_complex*'
local tools=require 'tools'

local complex = require 'complex'

function conv_lib.my_convolution_2d(x,filt,ds)
   assert(tools.is_complex(x),'The signal should be complex')
  assert(tools.start_with_equal_dimension(x,filt),'The signal should be of the same size')
   assert(x:size(1) % 2^ds ==0,'First dimension should be a multiple of 2^2ds')
   assert(x:size(2) % 2^ds ==0,'First dimension should be a multiple of 2^2ds')   
   local yf=complex.multiply_complex_tensor(x,filt)
   yf=conv_lib.periodize_along_k(conv_lib.periodize_along_k(yf,1,ds),2,ds)
   
   return conv_lib.my_fft_complex(conv_lib.my_fft_complex(yf,1,1),2,1)
end

function conv_lib.pad_signal(x,size_to_pad)
   return x
end



function conv_lib.periodize_along_k(h,k,l)
      assert(k<=h:nDimension(),'k is bigger than the dimension')
   local dim=h:size()
   
   local final_dim=torch.LongStorage(#dim)
   local new_dim=torch.LongStorage(#dim+1)
   for i=1,k-1 do
      new_dim[i]=dim[i]
      final_dim[i]=dim[i]
   end
   
   new_dim[k]=2^l
   final_dim[k]=dim[k]/2^l
   new_dim[k+1]=dim[k]/2^l
   for i=k+1,#dim do
      new_dim[i+1]=dim[i]
      final_dim[i]=dim[i]
   end
   
   local reshaped_h=torch.view(h,new_dim)
   
   local  summed_h=torch.view(torch.sum(reshaped_h,k),final_dim)
   return summed_h
end


return conv_lib