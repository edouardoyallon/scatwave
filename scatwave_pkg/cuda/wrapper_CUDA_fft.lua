local wrapper_CUDA_fft={}
local ffi=require 'ffi'
local fftw_complex_cast = 'fftw_complex*'
local fftw_dim_cast='fftw_iodim[?]'

local cuFFT=require 'cuda/engine_CUDA'

function wrapper_CUDA_fft.my_fft_complex(x,k,backward)   
   -- Defines the 1D transform along the dimension k using the stride of the tensor. There is no need to be contiguous along this dimension.
   local dims = ffi.new(fftw_dim_cast,1)
   dims[0] = {n=x:size(k),is=x:stride(k)/2,os=x:stride(k)/2}      
      
   -- Defines the elements along which we want to apply the transform previously defined.
   local howmany_dims = ffi.new(fftw_dim_cast,x:nDimension()-2)   
   local j = 0
   for l = 1, x:nDimension()-1 do
      if(l>k or l<k) then
         howmany_dims[j] = { n=x:size(l),is=x:stride(l)/2,os=x:stride(l)/2 }
         j = j+1
      end
   end

   local flags = cuFFT.ESTIMATE

   local in_data = torch.data(x)
   local in_data_cast = ffi.cast(fftw_complex_cast, in_data)
   
   -- Output data: size and stride are identical      
   local output = torch.Tensor(x:size(),x:stride()):zero()
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftw_complex_cast, output_data)
   
   -- iFFT if needed, no normalization is performed by FFTW3.0
   if not backward then
      sign = cuFFT.FORWARD
   else
      sign = cuFFT.BACKWARD
   end
   
   -- The plan!
   local plan = cuFFT.plan_guru_dft(1, dims, x:nDimension()-2, howmany_dims, in_data_cast, output_data_cast, sign, flags)
   
   -- If you observe a SEGFAULT, it can only come from this part.
   cuFFT.execute(plan)
   
    if(backward) then
      output=torch.div(output,x:size(k))   
   end
      return output
end

return wrapper_CUDA_fft