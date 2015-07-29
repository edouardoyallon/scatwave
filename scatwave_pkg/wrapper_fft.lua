local ffi=require 'ffi'
local fftw = require 'fftw3' -- this library has been implemented by soumith, we could avoid using it.
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

   local flags = fftwf.ESTIMATE

   local in_data = torch.data(x)
   local in_data_cast = ffi.cast(fftwf_complex_cast, in_data)
   
   local output = torch.FloatTensor(x:size(),x:stride()):zero()
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftwf_complex_cast, output_data)
   
   -- iFFT if needed, keep in mind that no normalization is performed by FFTW3.0
   if not backward then
      sign = fftwf.FORWARD
   else
      sign = fftwf.BACKWARD
   end
   
   local idist=x:size(k)*x:size(k+1)
   local istride=1
      
   local ostride=istride
   local odist=idist

   -- The plan!
   
   --[[-- SEGFAULT CAN COME FROM THOSE LINES ]]
   local plan = fftwf.plan_many_dft(2, dims, batch,in_data_cast, dims, istride,idist,output_data_cast, dims,istride,odist,sign, flags)
   fftwf.execute(plan)
   fftwf.destroy_plan(plan)   
   --[[--------------------------------------]]

   if(backward) then
      output=torch.div(output,x:size(k)*x:size(k))   
   end
   return output
end




function wrapper_fft.my_2D_fft_complex(x,backward)   
   local flags = fftwf.ESTIMATE

   local in_data = torch.data(x)
   local in_data_cast = ffi.cast(fftwf_complex_cast, in_data)
   
   -- Output data: size and stride are identical      
   local output = torch.FloatTensor(x:size(),x:stride()):zero()
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftwf_complex_cast, output_data)
   
   -- iFFT if needed, no normalization is performed by FFTW3.0
   if not backward then
      sign = fftwf.FORWARD
   else
      sign = fftwf.BACKWARD
   end

   -- The plan!
   
   --[[-- SEGFAULT CAN COME FROM THOSE LINES ]]
   local plan = fftwf.plan_dft_2d(x:size(1),x:size(2), in_data_cast, output_data_cast, sign,flags)
   fftwf.execute(plan)
   fftwf.destroy_plan(plan)   
   --[[--------------------------------------]]

   if(backward) then
      output=torch.div(output,x:size(k))   
   end
      return output
end


function wrapper_fft.my_2D_fft_real_batch(x,k)  
   -- Defines a 2D convolution that batches until the k-th dimension
   local dims = ffi.new('int[?]',2)
   dims[0] = x:size(k)
   dims[1] = x:size(k+1)
   
   local batch=x:nElement()/(x:size(k)*x:size(k+1))

   local flags = fftwf.ESTIMATE

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
      sign = fftwf.FORWARD
   else
      sign = fftwf.BACKWARD
   end
   
   local idist=x:size(k)*x:size(k+1)
   local istride=1
      
   local ostride=istride
   local odist=idist

   -- The plan!
   
   --[[-- SEGFAULT CAN COME FROM THOSE LINES ]]
   local plan = fftwf.plan_many_dft_r2c(2, dims, batch,in_data_cast, dims, istride,idist,output_data_cast, dims,istride,odist, flags)
   fftwf.execute(plan)
   fftwf.destroy_plan(plan)   
   --[[--------------------------------------]]
   
   k=k+1 -- We get the half of the coefficients that were not computed by fftwf ?? IS IT ?

--   local n_el=torch.floor((x:size(k)-1)/2)
--   local n_med=2+torch.floor((x:size(k))/2)
   
   -- If there is something to fill in ? IS IT ??
--   if(n_el>0) then
--      output:narrow(k,n_med,n_el):indexCopy(k,torch.range(n_el,1,-1):long(),output:narrow(k,2,n_el))
--      output:narrow(k,torch.ceil(x:size(k)/2)+1,torch.floor(x:size(k)/2)):narrow(output:nDimension(),2,1):mul(-1)
--   end
   return output
end   

return wrapper_fft
