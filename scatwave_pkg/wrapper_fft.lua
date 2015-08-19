local ffi=require 'ffi'
local fftw = require 'engine'
local fftwf_complex_cast = 'fftwf_complex*'
local fftw_dim_cast='fftw_iodim[?]'
local tools=require 'tools'
local fftwf=fftw.float
local complex = require 'complex'

local wrapper_fft={}


function wrapper_fft.my_2D_fft_complex_batch(x,k,backward)   
   -- Defines a 2D convolution that batches until the k-th dimension
   local dims = ffi.new('int[?]',2)
   dims[0] = x:size(k)
   dims[1] = x:size(k+1)
   
   local batch=x:nElement()/(2*x:size(k)*x:size(k+1))

   local flags = fftw.ESTIMATE

   local in_data = torch.data(x)
   local in_data_cast = ffi.cast(fftwf_complex_cast, in_data)
   
   local output = torch.FloatTensor(x:size(),x:stride()):zero()
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftwf_complex_cast, output_data)
   
   -- iFFT if needed, keep in mind that no normalization is performed by FFTW3.0
   if not backward then
      sign = fftw.FORWARD
   else
      sign = fftw.BACKWARD
   end
   
   local idist=x:size(k)*x:size(k+1)
   local istride=1
      
   local ostride=istride
   local odist=idist

   -- The plan!
   
   --[[-- SEGFAULT CAN COME FROM THOSE LINES ]]
   local plan =    fftw.C['fftwf_plan_many_dft'](2, dims, batch,in_data_cast, dims, istride,idist,output_data_cast, dims,istride,odist,sign, flags)
   fftw.C['fftwf_execute'](plan)
   fftw.C['fftwf_destroy_plan'](plan)   
   --[[--------------------------------------]]

   if(backward) then
      output:div(x:size(k)*x:size(k+1))   
   end
   return output
end




function wrapper_fft.my_2D_fft_complex(x,backward)   
   local flags = fftw.ESTIMATE

   local in_data = torch.data(x)
   local in_data_cast = ffi.cast(fftwf_complex_cast, in_data)
   
   -- Output data: size and stride are identical      
   local output = torch.FloatTensor(x:size(),x:stride()):zero()
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftwf_complex_cast, output_data)
   
   -- iFFT if needed, no normalization is performed by FFTW3.0
   if not backward then
      sign = fftw.FORWARD
   else
      sign = fftw.BACKWARD
   end

   -- The plan!
   
   --[[-- SEGFAULT CAN COME FROM THOSE LINES ]]
   local plan =    fftw.C['fftwf_plan_dft_2d'](x:size(1),x:size(2), in_data_cast, output_data_cast, sign,flags)
   fftw.C['fftwf_execute'](plan)
      fftw.C['fftwf_destroy_plan'](plan)   
   --[[--------------------------------------]]

   if(backward) then
      output:div(x:size(1)*x:size(2))   
   end
      return output
end


function wrapper_fft.my_2D_fft_real_batch(x,k)  
   -- Defines a 2D convolution that batches until the k-th dimension
   local dims = ffi.new('int[?]',2)
   dims[0] = x:size(k)
   dims[1] = x:size(k+1)
   
   local batch=x:nElement()/(x:size(k)*x:size(k+1))

   local flags = fftw.ESTIMATE

   local in_data = torch.data(x)
   local in_data_cast = ffi.cast('float*', in_data)
   
   
   local fs = torch.LongStorage(x:nDimension()+1)
   for l = 1, x:nDimension() do
      fs[l] = x:size(l)
   end
   fs[x:nDimension()+1] = 2
 
   local ss = torch.LongStorage(x:nDimension()+1)
   for l = 1, x:nDimension() do
      ss[l] = x:stride(l)*2
   end
   ss[x:nDimension()+1]=1
   
   


   local output = torch.FloatTensor(fs,ss):zero()
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftwf_complex_cast, output_data)
   
   -- iFFT if needed, keep in mind that no normalization is performed by FFTW3.0
   if not backward then
      sign = fftw.FORWARD
   else
      sign = fftw.BACKWARD
   end
   
   local idist=x:size(k)*x:size(k+1)
   local istride=1
      
   local ostride=istride
   local odist=idist

   -- The plan!
   
   --[[-- SEGFAULT CAN COME FROM THOSE LINES ]]
   local plan = fftw.C['fftwf_plan_many_dft_r2c'](2, dims, batch,in_data_cast, dims, istride,idist,output_data_cast, dims,istride,odist, flags)
   fftw.C['fftwf_execute'](plan)
      fftw.C['fftwf_destroy_plan'](plan)  
   --[[--------------------------------------]]
      
   local n_el=x:size(k)-1
   local n_med=2
   
   local n_el_kp1=torch.floor((x:size(k+1)-1)/2)
   local n_med_kp1=2+torch.floor((x:size(k+1))/2)  
   
   -- If there is something to fill in ? IS IT ??
   if(n_el>0) then
      local subs_idx=output:narrow(k+1,n_med_kp1,n_el_kp1)
      subs_idx:indexCopy(k+1,torch.range(n_el_kp1,1,-1):long(),output:narrow(k+1,2,n_el_kp1))
      local subs_subs_idx=subs_idx:narrow(k,n_med,n_el)
      
      local tmp=torch.FloatTensor(subs_subs_idx:size(),subs_subs_idx:stride()) -- hard to avoid, because there are some funny memory conflicts...
      tmp:indexCopy(k,torch.range(x:size(k)-1,1,-1):long(),subs_subs_idx)
      subs_subs_idx:copy(tmp)
      
      -- conjugate         
      output:narrow(k+1,n_med_kp1,n_el_kp1):narrow(output:nDimension(),2,1):mul(-1)
   end


   return output
end   

return wrapper_fft
