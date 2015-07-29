local ffi=require 'ffi'
local fftw = require 'fftw3' -- this library has been implemented by soumith, we could avoid using it.
local fftw_complex_cast = 'fftw_complex*'
local fftw_dim_cast='fftw_iodim[?]'
local tools=require 'tools'

local complex = require 'complex'

local my_fft={}

-- CUDA functions are missing!

function my_fft.my_fft_real(x,k)
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

   local flags = fftw.ESTIMATE

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
      
   local output = torch.Tensor(fs,ss):zero()
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftw_complex_cast, output_data)
      
   -- The plan!
   local plan = fftw.plan_guru_dft_r2c(1,dims, x:nDimension()-1, howmany_dims, in_data_cast, output_data_cast, flags)
   
   -- If you observe a SEGFAULT, it can only come from this part.
   fftw.execute(plan)
   
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


function my_fft.my_fft_complex(x,k,backward)   
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

   local flags = fftw.ESTIMATE

   local in_data = torch.data(x)
   local in_data_cast = ffi.cast(fftw_complex_cast, in_data)
   
   -- Output data: size and stride are identical      
   local output = torch.Tensor(x:size(),x:stride()):zero()
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftw_complex_cast, output_data)
   
   -- iFFT if needed, no normalization is performed by FFTW3.0
   if not backward then
      sign = fftw.FORWARD
   else
      sign = fftw.BACKWARD
   end
   
   -- The plan!
   local plan = fftw.plan_guru_dft(1, dims, x:nDimension()-2, howmany_dims, in_data_cast, output_data_cast, sign, flags)
   
   -- If you observe a SEGFAULT, it can only come from this part.
   fftw.execute(plan)
   
    if(backward) then
      output=torch.div(output,x:size(k))   
   end
      return output
end

return my_fft
