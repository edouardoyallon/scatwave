local conv_lib={}
local tools=require 'tools'
--local ffi=require 'ffi'
--local fftw = require 'fftw3'
--local fftw_complex_cast = 'fftw_complex*'
local tools=require 'tools'
--local my_fft = require 'my_fft'

local complex = require 'complex'

function conv_lib.my_convolution_2d(x,filt,ds,mini_batch,my_fft)
   assert(tools.is_complex(x),'The signal should be complex')
--   assert(tools.are_equal_dimension(x,filt),'The signal should be of the same size')
   assert(x:size(1+mini_batch) % 2^ds ==0,'First dimension should be a multiple of 2^2ds')
   assert(x:size(2+mini_batch) % 2^ds ==0,'First dimension should be a multiple of 2^2ds')   
   local yf=complex.multiply_complex_tensor(x,filt,mini_batch)
   yf=conv_lib.periodize_along_k(conv_lib.periodize_along_k(yf,1+mini_batch,ds),2+mini_batch,ds)
   return my_fft.my_fft_complex(my_fft.my_fft_complex(yf,1+mini_batch,1),2+mini_batch,1)
end


-- Apply a symetric padding and center the signal-- assuming the signal is real!
function conv_lib.pad_signal_along_k(x,new_size_along_k,k)
   local fs=torch.LongStorage(x:nDimension())
   for l=1,x:nDimension() do
      if(l==k) then
         fs[l]=new_size_along_k
      else
         fs[l]=x:size(l)
      end
   end
   
   local id_tmp = torch.cat(torch.range(1,x:size(k),1):long(),torch.range(x:size(k),1,-1):long())
   
   local idx=torch.LongTensor(new_size_along_k)
   local n_decay=torch.floor((new_size_along_k-x:size(k))/2)
   for i=1,new_size_along_k do
      idx[i]=(id_tmp[(i-n_decay-1)%id_tmp:size(1)+1])
   end   

   local y=torch.Tensor(fs)
   y:index(x,k,idx) 
   return y
end


function conv_lib.unpad_signal_along_k(x,original_size_along_k,k,res) 
   local n_decay=torch.floor((x:size(k)*2^res-original_size_along_k)/2^(res+1))+1   
   local f_size_along_k=1+torch.floor((original_size_along_k-1)/(2^res))
   local y=torch.Tensor.narrow(x,k,n_decay,f_size_along_k)
   return y
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
   summed_h:mul(1/2^l)
   return summed_h
end



return conv_lib