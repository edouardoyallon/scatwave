local wrapper_CUDA_fft={}
local ffi=require 'ffi'
local fftwf_complex_cast = 'fftwf_complex*'

local cuFFT=require 'cuda/engine_CUDA_nvidia'


function wrapper_CUDA_fft.my_2D_fft_complex_batch(x,k,backward)   
   -- Defines a 2D convolution that batches until the k-th dimension
   local n = ffi.new('int[?]',2)
   n[0] = x:size(k)
   n[1] = x:size(k+1)
   local batch=x:nElement()/(2*x:size(k)*x:size(k+1))
   local idist = x:size(k)*x:size(k+1)
   local istride = 1
      
   local ostride = istride
   local odist = idist
   local rank = 2
   local type = cuFFT.C2C
   -- The plan!  
   --[[-- SEGFAULT CAN COME FROM THOSE LINES ]]
   
   local plan_cast = ffi.new('int[?]',1)
        
   result=cuFFT.planMany(plan_cast, rank,n, n, istride, idist, n, ostride, odist, type, batch)
   cutorch.synchronize()
   print(plan_cast[0])
   print(result)
cuFFT.destroy(plan_cast[0])         
--[[--   print(result)
   local in_data = torch.data(x)
   local in_data_cast = ffi.cast(fftwf_complex_cast, in_data)
   

   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftwf_complex_cast, output_data)                  
                  
                  
      -- iFFT if needed, keep in mind that no normalization is performed by FFTW3.0
   if not backward then
      sign = cuFFT.FORWARD
   else
      sign = cuFFT.BACKWARD
   end            ]]--
                  
                  
                  
                  
                  
--        cutorch.synchronize()      
--print(plan)
--   cuFFT.destroy_plan(plan)         
--print(plan)

   

--cutorch.synchronize()   
--   cuFFT.execute(plan)

   local output = torch.CudaTensor(x:size(),x:stride()):zero()
   --[[--------------------------------------]]

   if(backward) then
      output=torch.div(output,x:size(k))   
   end
   return output
end




function wrapper_CUDA_fft.my_2D_fft_complex(x,backward)   
   local flags = cuFFT.ESTIMATE

   local in_data = torch.data(x)
   local in_data_cast = ffi.cast(fftwf_complex_cast, in_data)
   
   -- Output data: size and stride are identical      
   local output = torch.CudaTensor(x:size(),x:stride()):zero()
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftwf_complex_cast, output_data)
   
   -- iFFT if needed, no normalization is performed by FFTW3.0
   if not backward then
      sign = cuFFT.FORWARD
   else
      sign = cuFFT.BACKWARD
   end

   -- The plan!
   --[[-- SEGFAULT CAN COME FROM THOSE LINES ]]
--   local plan = cuFFT.plan_dft_2d(x:size(1),x:size(2), in_data_cast, output_data_cast, sign,flags)
--   cuFFT.execute(plan)
--      cutorch.synchronize()
--   cuFFT.destroy_plan(plan)   
   --[[--------------------------------------]]

   if(backward) then
      output=torch.div(output,x:size(k))   
   end
      return output
end


function wrapper_CUDA_fft.my_2D_fft_real_batch(x,k)   
   -- Defines a 2D convolution that batches until the k-th dimension
   local dims = ffi.new('int[?]',2)
   dims[0] = x:size(k)
   dims[1] = x:size(k+1)
   
   local batch=x:nElement()/(x:size(k)*x:size(k+1))

   local flags = cuFFT.ESTIMATE

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

   local output = torch.CudaTensor(fs,ss):zero()
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftwf_complex_cast, output_data)
   
   -- iFFT if needed, keep in mind that no normalization is performed by FFTW3.0
   if not backward then
      sign = cuFFT.FORWARD
   else
      sign = cuFFT.BACKWARD
   end
   
   local idist=x:size(k)*x:size(k+1)
   local istride=1
      
   local ostride=istride
   local odist=idist

   -- The plan!
--[[-- SEGFAULT CAN COME FROM THOSE LINES ]]
--   local plan = cuFFT.plan_many_dft_r2c(2, dims, batch,in_data_cast, dims, istride,idist,output_data_cast, dims,istride,odist, flags)
--   cuFFT.execute(plan)
--   cuFFT.destroy_plan(plan)   
   --[[--------------------------------------]]
   
   
  -- k=k+2 -- We get the half of the coefficients that were not computed by cuFFT - SHOULD WE REMOVE INDEED THOSE LINES? NEED TO BE CHECKED!

   --local n_el=torch.floor((x:size(k)-1)/2)
--   local n_med=2+torch.floor((x:size(k))/2)
   
   -- If there is something to fill in
  -- if(n_el>0) then
    --  output:narrow(k,n_med,n_el):indexCopy(k,torch.range(n_el,1,-1):long(),output:narrow(k,2,n_el))
   
      --output:narrow(k,torch.ceil(x:size(k)/2)+1,torch.floor(x:size(k)/2)):narrow(output:nDimension(),2,1):mul(-1)
--      end
   return output
end   
   
return wrapper_CUDA_fft
