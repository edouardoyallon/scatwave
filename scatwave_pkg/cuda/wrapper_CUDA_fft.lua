local wrapper_CUDA_fft={}
local ffi=require 'ffi'
local fftwf_complex_cast = 'fftwf_complex*'
local fftw_dim_cast='fftw_iodim[?]'

local cuFFT=require 'cuda/engine_CUDA'


function wrapper_CUDA_fft.my_2D_fft_complex_batch(x,k,backward)   
   -- Defines the 1D transform along the dimension k using the stride of the tensor. There is no need to be contiguous along this dimension.
   local dims = ffi.new('int[?]',2)
   dims[0] = x:size(k)--{n=x:size(k),is=x:stride(k)/2,os=x:stride(k)/2}      
dims[1] = x:size(k+1)
   
   local batch=x:nElement()/(2*x:size(k)*x:size(k+1))

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
   
   local idist=x:size(k)*x:size(k+1)
   local istride=1
      
   local ostride=istride
   local odist=idist


   -- The plan!

local plan = cuFFT.plan_many_dft(2, dims, batch,in_data_cast, dims, istride,idist,output_data_cast, dims,istride,odist,sign, flags)
--print(plan)
 --local plan = cuFFT.plan_dft_2d(x:size(1),x:size(2), in_data_cast, output_data_cast, sign,flags)
   -- If you observe a SEGFAULT, it can only come from this part.
cuFFT.execute(plan)
  cuFFT.destroy_plan(plan)   

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
   local plan = cuFFT.plan_dft_2d(x:size(1),x:size(2), in_data_cast, output_data_cast, sign,flags)
   -- If you observe a SEGFAULT, it can only come from this part.
   cuFFT.execute(plan)
   cuFFT.destroy_plan(plan)   

   if(backward) then
      output=torch.div(output,x:size(k))   
   end
      return output
end


function wrapper_CUDA_fft.my_2D_fft_real(x,k)   
   -- Defines the 1D transform along the dimension k using the stride of the tensor. There is no need to be contiguous along this dimension.
   local dims=ffi.new(fftw_dim_cast,1)
   dims[0] = {n=x:size(k),is=x:stride(k),os=x:stride(k)}
   
   -- Defines the elements along which we want to apply the transform previously defined.
   local howmany_dims = ffi.new(fftw_dim_cast,x:nDimension()-1)   
   local j = 0
   for l=1,x:nDimension() do
      if(l>k or l<k) then
         howmany_dims[j] = { n = x:size(l), is = x:stride(l),os = x:stride(l) }
         j = j+1
      end
   end

   local flags = cuFFT.ESTIMATE

   -- Output data: we copy the size and the stride
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

   local in_data = torch.data(x)
   local in_data_cast = ffi.cast('double*', in_data)
      
   local output = torch.CudaTensor(fs,ss):zero()
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftwf_complex_cast, output_data)
      
   -- The plan!
   --local plan = cuFFT.plan_guru_dft_r2c(1,dims, x:nDimension()-1, howmany_dims, in_data_cast, output_data_cast, flags)
   
--local plan = cuFFT.plan_dft_2d(10,10, in_data_cast, output_data_cast, 1,flags)
   
   print(plan)

   -- If you observe a SEGFAULT, it can only come from this part.
  -- cuFFT.execute(plan)
   
   -- Since the returned output takes in account the symetry and avoid redundancy, we need to manually add them
   local n_el=torch.floor((x:size(k)-1)/2)
   local n_med=2+torch.floor((x:size(k))/2)
   
   -- If there is something to fill in
   if(n_el>0) then
      output:narrow(k,n_med,n_el):indexCopy(k,torch.range(n_el,1,-1):long(),output:narrow(k,2,n_el))
   
      output:narrow(k,torch.ceil(x:size(k)/2)+1,torch.floor(x:size(k)/2)):narrow(output:nDimension(),2,1):mul(-1)
   end
      return output
end



return wrapper_CUDA_fft