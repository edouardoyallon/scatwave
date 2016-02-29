--[[
     ScatWave implementation of Scattering Network
     Written by Edouard Oyallon
     Team DATA ENS
     Copyright 2015
  ]]

local wrapper_CUDA_fft={}
local ffi=require 'ffi'
local fftwf_complex_cast = 'fftwf_complex*'

local cuFFT=require 'scatwave.cuda/engine_CUDA_nvidia'


wrapper_CUDA_fft.LUT={}
wrapper_CUDA_fft.TMP={}


local function destroy_plan(p)
   cuFFT.C['cufftDestroy'](p[0])      
      
end



function wrapper_CUDA_fft.cache(x,k,type)
   local plan_cast = ffi.new('int[1]',{})
   local n = ffi.new('int[2]',{})
   n[0] = x:size(k)
   n[1] = x:size(k+1)
   local batch
   if(type==cuFFT.R2C) then
      batch=x:nElement()/(x:size(k)*x:size(k+1))
   else
      batch=x:nElement()/(2*x:size(k)*x:size(k+1))
   end
   local idist = x:size(k)*x:size(k+1)
   local istride = 1
   
   local ostride = istride
   local odist = idist
   local rank = 2
   
   cuFFT.C['cufftPlanMany'](plan_cast, rank,n, n, istride, idist, n, ostride, odist, type, batch)   
      ffi.gc(plan_cast,destroy_plan)
   return plan_cast
end



function wrapper_CUDA_fft.my_2D_fft_complex_batch(x,k,backward,out)   
   -- Defines a 2D convolution that batches until the k-th dimension
   
   local type = cuFFT.C2C
   local batch=x:nElement()/(2*x:size(k)*x:size(k+1))
   -- The plan!  
   
   local in_data = torch.data(x)
   local in_data_cast = ffi.cast(fftwf_complex_cast, in_data)
   
   local output
   if(not out) then
      output = torch.CudaTensor(x:size(),x:stride()):zero()
   else
      output=out
   end
   
   
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftwf_complex_cast, output_data)                  
      
      
   local sign
      -- iFFT if needed, keep in mind that no normalization is performed by FFTW3.0
   if not backward then
      sign = cuFFT.FORWARD
   else
      sign = cuFFT.INVERSE
   end          
   
   
--[[-- SEGFAULT CAN COME FROM THOSE LINES ]]
   if(wrapper_CUDA_fft.LUT[batch]==nil) then
      wrapper_CUDA_fft.LUT[batch]={}
   end
   if(wrapper_CUDA_fft.LUT[batch][x:size(k)]==nil) then
      wrapper_CUDA_fft.LUT[batch][x:size(k)]={}
   end
   if(wrapper_CUDA_fft.LUT[batch][x:size(k)][type]==nil) then
      wrapper_CUDA_fft.LUT[batch][x:size(k)][type]=wrapper_CUDA_fft.cache(x,k,type)
   end
   
   local plan_cast=wrapper_CUDA_fft.LUT[batch][x:size(k)][type][0]
   
   cuFFT.C['cufftExecC2C'](plan_cast,in_data_cast,output_data_cast,sign)              
      
--[[--------------------------------------]]
      
   if(backward) then
      output:div(x:size(k)*x:size(k+1))   
   end
   return output
end



-- IT DESTROYS THE INPUT!(i.e. x)
function wrapper_CUDA_fft.my_2D_ifft_complex_to_real_batch(x,k,out)
   
   
   local in_data = torch.data(x)
   local in_data_cast = ffi.cast(fftwf_complex_cast, in_data)
   
   local batch=x:nElement()/(2*x:size(k)*x:size(k+1))
   
   local type = cuFFT.C2R
   
   local in_data = torch.data(x)
   local in_data_cast = ffi.cast(fftwf_complex_cast, in_data)
   
   
   local fs = torch.LongStorage(x:nDimension()-1)
   for l = 1, x:nDimension()-1 do
      fs[l] = x:size(l)
   end
   
   local output   
   if(not out) then
      output = torch.CudaTensor(fs):zero()
   else
      output = out
   end
   
   
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast('float*', output_data)
   
   -- The plan!
   
--[[-- SEGFAULT CAN COME FROM THOSE LINES ]]
   if(wrapper_CUDA_fft.LUT[batch]==nil) then
      wrapper_CUDA_fft.LUT[batch]={}
   end
   if(wrapper_CUDA_fft.LUT[batch][x:size(k)]==nil) then
      wrapper_CUDA_fft.LUT[batch][x:size(k)]={}
   end
   if(wrapper_CUDA_fft.LUT[batch][x:size(k)][type]==nil) then
      wrapper_CUDA_fft.LUT[batch][x:size(k)][type]=wrapper_CUDA_fft.cache(x,k,type)
   end
   local plan_cast=wrapper_CUDA_fft.LUT[batch][x:size(k)][type][0]
   
   
   cuFFT.C['cufftExecC2R'](plan_cast,in_data_cast,output_data_cast)
--[[--------------------------------------]]
   
   
   output:div(x:size(k)*x:size(k+1))   
      
   return output
end




function wrapper_CUDA_fft.my_2D_fft_complex(x,backward,out)   
   
   local in_data = torch.data(x)
   local in_data_cast = ffi.cast(fftwf_complex_cast, in_data)
   
   -- Output data: size and stride are identical      
   local output 
   if(not out) then
      output = torch.CudaTensor(x:size(),x:stride()):zero()
   else
      output=out
   end   
   
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftwf_complex_cast, output_data)
   
   local type = cuFFT.C2C   
      
      -- The plan!  
   local plan_cast = ffi.new('int[1]',{})
   
   
   local sign
      -- iFFT if needed, no normalization is performed by FFTW3.0
   if not backward then
      sign = cuFFT.FORWARD
   else
      sign = cuFFT.INVERSE
   end
   
--[[-- SEGFAULT CAN COME FROM THOSE LINES ]]
   cuFFT.C['cufftPlan2D'](plan_cast,x:size(1),x:size(2),type)
   cuFFT.C['cufftExecC2C'](plan_cast[0],in_data_cast,output_data_cast,sign)              
      cuFFT.C['cufftDestroy'](plan_cast[0])      
--[[--------------------------------------]]
      
   if(backward) then
      output=torch.div(output,x:size(k))   
   end
   return output
end


function wrapper_CUDA_fft.my_2D_fft_real_batch(x,k,out)   
   -- Defines a 2D convolution that batches until the k-th dimension
   
   local batch=x:nElement()/(x:size(k)*x:size(k+1))
   local type = cuFFT.R2C
   
   
   
   local in_data = torch.data(x)
   local in_data_cast = ffi.cast('float*', in_data)
   
   local fs = torch.LongStorage(x:nDimension()+1)
   for l = 1, x:nDimension() do
      fs[l] = x:size(l)
   end
   
   
   fs[x:nDimension()+1] = 2
   local output 
   if(not out) then
      output = torch.CudaTensor(fs):zero()
   else
      output=out
   end
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftwf_complex_cast, output_data)
   
   
   
   -- iFFT if needed, keep in mind that no normalization is performed by FFTW3.0
   if not backward then
      sign = cuFFT.FORWARD
   else
      sign = cuFFT.INVERSE
   end
   
   -- The plan!
   
--[[-- SEGFAULT CAN COME FROM THOSE LINES ]]
   if(wrapper_CUDA_fft.LUT[batch]==nil) then
      wrapper_CUDA_fft.LUT[batch]={}
   end
   if(wrapper_CUDA_fft.LUT[batch][x:size(k)]==nil) then
      wrapper_CUDA_fft.LUT[batch][x:size(k)]={}
   end
   if(wrapper_CUDA_fft.LUT[batch][x:size(k)][type]==nil) then
      wrapper_CUDA_fft.LUT[batch][x:size(k)][type]=wrapper_CUDA_fft.cache(x,k,type)
   end
   local plan_cast=wrapper_CUDA_fft.LUT[batch][x:size(k)][type][0]
   
   
   cuFFT.C['cufftExecR2C'](plan_cast,in_data_cast,output_data_cast)
--[[--------------------------------------]]
   
   local n_el=x:size(k)-1
   local n_med=2
   
   local n_el_kp1=torch.floor((x:size(k+1)-1)/2)
   local n_med_kp1=2+torch.floor((x:size(k+1))/2)  
      
   if(n_el>0 and n_el_kp1>0) then
      local subs_idx=output:narrow(k+1,n_med_kp1,n_el_kp1)
      subs_idx:indexCopy(k+1,torch.range(n_el_kp1,1,-1):long(),output:narrow(k+1,2,n_el_kp1))
      local subs_subs_idx=subs_idx:narrow(k,n_med,n_el)

      -- We cache the copy
      if(wrapper_CUDA_fft.TMP[batch]==nil) then
         wrapper_CUDA_fft.TMP[batch]={}
      end
      if(wrapper_CUDA_fft.TMP[batch][x:size(k)]==nil) then
         wrapper_CUDA_fft.TMP[batch][x:size(k)]={}
      end
      if(wrapper_CUDA_fft.TMP[batch][x:size(k)]) then
         wrapper_CUDA_fft.TMP[batch][x:size(k)]=torch.CudaTensor(subs_subs_idx:size(),subs_subs_idx:stride())
      end      
      local tmp=wrapper_CUDA_fft.TMP[batch][x:size(k)]       -- unavoidable because of the flip.

         tmp:indexCopy(k,torch.range(x:size(k)-1,1,-1):long(),subs_subs_idx)
      subs_subs_idx:copy(tmp)
      
      -- conjugate         
      output:narrow(k+1,n_med_kp1,n_el_kp1):narrow(output:nDimension(),2,1):mul(-1)
   end

   return output
end   

return wrapper_CUDA_fft
