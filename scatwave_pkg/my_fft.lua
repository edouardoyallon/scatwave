local ffi=require 'ffi'
local fftw = require 'fftw3'
local fftw_complex_cast = 'fftw_complex*'
local tools=require 'tools'

local complex = require 'complex'

local my_fft={}

-- CUDA functions for the FFT... NOT IMPLEMENTED YET! :-(
function my_fft.my_fft_real_2D_cuda(x,k)   
   
end


function my_fft.my_fft_real(x,k)   
   assert(not tools.is_complex(x),'Signal is not real')
   assert(k<=x:nDimension(),'k is bigger than the dimension')
   local input=x
   
   --- Parameters of the fft PLAN
   local rank=1
   
   local n=torch.LongTensor({{x:size(k)}})
   local n_data=torch.data(n)
   local n_data_cast= ffi.cast('const int*',n_data)

   
   local howmany=x:nElement()/x:size(k)  

      
   input = input:contiguous() -- make sure input is contiguous
   local input_data = torch.data(input)
   local input_data_cast = ffi.cast('double*', input_data)
   
   local inembed_data_cast=n_data_cast--ffi.cast('const int*',0)
      
   local istride=x:stride(k)
   local idist=x:nElement()/(x:stride(k)*x:size(k))



   local fs=torch.LongStorage(x:nDimension()+1)
   for l=1,x:nDimension() do
      fs[l]=x:size(l)
   end
   fs[x:nDimension()+1]=2
   
   local output = torch.Tensor(fs):typeAs(input):zero()
   
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftw_complex_cast, output_data)
   
   local oembed_data_cast=n_data_cast--ffi.cast('const int*',oembed_data)
      
   local ostride=istride
   local odist=idist
   
   
   local flags = fftw.ESTIMATE
   
   local plan  = fftw.plan_many_dft_r2c(rank,n_data_cast,howmany,input_data_cast,inembed_data_cast,
                                        istride,idist,output_data_cast,oembed_data_cast,ostride,odist,flags)   

    fftw.execute(plan)
   
   
   local n_el=   torch.floor((x:size(k)-1)/2)
   local n_med= 2+torch.floor((x:size(k))/2)
   
   
   output:narrow(k,n_med,n_el):indexCopy(k,torch.range(n_el,1,-1):long(),output:narrow(k,2,n_el))
   
   output:narrow(k,torch.ceil(x:size(k)/2)+1,torch.floor(x:size(k)/2)):narrow(output:nDimension(),2,1):mul(-1)
   
   fftw.destroy_plan(plan)
   
   return output
end


function my_fft.my_fft_complex(x,k,backward) 
   
   assert(tools.is_complex(x),'Signal is not complex')
   assert(k<x:nDimension(),'k is bigger than the dimension')  
   local input
   local flag=backward or false      
      input=x
   local rank=1
   
   local n=torch.LongTensor({{x:size(k)}})
   local n_data=torch.data(n)
   local n_data_cast= ffi.cast('const int*',n_data)
   
   
   local howmany=x:nElement()/(2*x:size(k))   
      
      input = input:contiguous() -- make sure input is contiguous
   local input_data = torch.data(input)
   local input_data_cast = ffi.cast(fftw_complex_cast, input_data)
   
   local inembed_data_cast=n_data_cast--ffi.cast('const int*',0)
   
   local idist=x:nElement()/(x:stride(k)*x:size(k)) 
      
   local istride=x:stride(k)/2
   
   
   local output = torch.Tensor(input:size()):typeAs(input):zero()
   
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftw_complex_cast, output_data)
   
   --  local oembed=nil
   --   local oembed_data=torch.data(oembed)
   local oembed_data_cast=n_data_cast--ffi.cast('const int*',oembed_data)
      
   local ostride=istride
   local odist=idist
   
   local sign
   if not backward then
      sign=fftw.FORWARD
   else
      sign=fftw.BACKWARD
   end
   
   local flags = fftw.ESTIMATE
   

   local plan  = fftw.plan_many_dft(rank,n_data_cast,howmany,input_data_cast,inembed_data_cast,
                                    istride,idist,output_data_cast,oembed_data_cast,ostride,odist,sign,flags)   
      
      fftw.execute(plan)
   
   fftw.destroy_plan(plan)
   
   if(flag) then
      output=torch.div(output,x:size(k))   
   end
   
   return output
end

return my_fft