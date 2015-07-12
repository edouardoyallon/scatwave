local ffi=require 'ffi'
local fftw = require 'fftw3'
local fftw_complex_cast = 'fftw_complex*'
local fftw_dim_cast='fftw_iodim[?]'
local tools=require 'tools'

local complex = require 'complex'

local my_fft={}

-- CUDA functions for the FFT... NOT IMPLEMENTED YET! :-(



function my_fft.my_fft_real(x,k)   
   local rank=x:nDimension() 
   local dims=ffi.new(fftw_dim_cast,1)
   print(x:stride(1))
   dims[0]={n=x:size(k),is=x:stride(k),os=x:stride(k)}--.n=x:size(k)--size of the dimension
--   dims[0].is=x:stride(k)-- input stride
 --  dims[0].os=x:stride(k)--output stride
      
   local dims_data = dims--torch.data(dims)
   local dims_cast = dims--ffi.cast(fftw_dim_cast, dims_data)

   local howmany_dims=ffi.new(fftw_dim_cast,x:nDimension()-1)   
   local howmany_rank=x:nDimension()-1

   local j=0
   for l=1,x:nDimension() do
      if(l>k or l<k) then
         print(x:stride(l))
         howmany_dims[j]={n=x:size(l),is=x:stride(l),os=x:stride(l)}
         j=j+1
      end
   end
   
   local howmany_dims_data = howmany_dims--torch.data(howmany_dims)
   local howmany_dims_cast = howmany_dims--ffi.cast(fftw_dim_cast, howmany_dims_data)

   local flags = fftw.ESTIMATE

   -- Output data
   
local fs=torch.LongStorage(x:nDimension()+1)
   for l=1,x:nDimension() do
      fs[l]=x:size(l)
   end
   fs[x:nDimension()+1]=2
   
   local ss=torch.LongStorage(x:nDimension()+1)
   for l=1,x:nDimension() do
      ss[l]=x:stride(l)*2
   end
   ss[x:nDimension()+1]=1

   local in_data = torch.data(x)
   local in_data_cast = ffi.cast('double*', in_data)
      
   local output=torch.Tensor(fs,ss):zero()
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftw_complex_cast, output_data)
   
local plan  = fftw.plan_guru_dft_r2c(
          1, dims_cast,
          x:nDimension()-1, howmany_dims_cast,
          in_data_cast,output_data_cast,
          flags)
  --[[-- local ro=output:select(output:nDimension(),1)
   local io=output:select(output:nDimension(),2)

   local ro_data=torch.data(ro)
   local io_data=torch.data(io)
   
   local io_data_cast = ffi.cast('double*', io_data) 
   local ro_data_cast = ffi.cast('double*', ro_data) 

   local plan  = fftw.plan_guru_dft_r2c(
          1, dims_cast,
          x:nDimension()-1, howmany_dims_cast,
          in_data_cast,ro_data_cast, io_data_cast,
          flags)--]]
   
   print(howmany_dims)
   print(plan)----- WHY DO YOU RETURN NULL
      fftw.execute(plan)
   
   
   
   local n_el=   torch.floor((x:size(k)-1)/2)
   local n_med= 2+torch.floor((x:size(k))/2)
   
   
   if(n_el>0) then
   output:narrow(k,n_med,n_el):indexCopy(k,torch.range(n_el,1,-1):long(),output:narrow(k,2,n_el))
   
      output:narrow(k,torch.ceil(x:size(k)/2)+1,torch.floor(x:size(k)/2)):narrow(output:nDimension(),2,1):mul(-1)
end
   
   
   
   
   

  -- return torch.cat(ro,io,x:nDimension()+1)
      return output

end





function my_fft.my_fft_complex(x,k,backward)   
   local rank=x:nDimension() 
   local dims=ffi.new(fftw_dim_cast,1)
   print(x:stride(1))
   dims[0]={n=x:size(k),is=x:stride(k)/2,os=x:stride(k)/2}--.n=x:size(k)--size of the dimension
--   dims[0].is=x:stride(k)-- input stride
 --  dims[0].os=x:stride(k)--output stride
      
   local dims_data = dims--torch.data(dims)
   local dims_cast = dims--ffi.cast(fftw_dim_cast, dims_data)

   local howmany_dims=ffi.new(fftw_dim_cast,x:nDimension()-2)   
   local howmany_rank=x:nDimension()-1

   local j=0
   for l=1,x:nDimension()-1 do
      if(l>k or l<k) then
         print(x:stride(l))
         howmany_dims[j]={n=x:size(l),is=x:stride(l)/2,os=x:stride(l)/2}
         j=j+1
      end
   end
   
   local howmany_dims_data = howmany_dims--torch.data(howmany_dims)
   local howmany_dims_cast = howmany_dims--ffi.cast(fftw_dim_cast, howmany_dims_data)

   local flags = fftw.ESTIMATE

   -- Output data
   

   local in_data = torch.data(x)
   local in_data_cast = ffi.cast(fftw_complex_cast, in_data)
      
   local output=torch.Tensor(x:size(),x:stride()):zero()
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftw_complex_cast, output_data)
   
   
   if not backward then
print('lolol')
      sign=fftw.FORWARD
   else
      sign=fftw.BACKWARD
   end

local plan  = fftw.plan_guru_dft(
          1, dims_cast,
          x:nDimension()-2, howmany_dims_cast,
          in_data_cast,output_data_cast,sign,
          flags)
  --[[-- local ro=output:select(output:nDimension(),1)
   local io=output:select(output:nDimension(),2)

   local ro_data=torch.data(ro)
   local io_data=torch.data(io)
   
   local io_data_cast = ffi.cast('double*', io_data) 
   local ro_data_cast = ffi.cast('double*', ro_data) 

   local plan  = fftw.plan_guru_dft_r2c(
          1, dims_cast,
          x:nDimension()-1, howmany_dims_cast,
          in_data_cast,ro_data_cast, io_data_cast,
          flags)--]]
   
   print(howmany_dims)
   print(plan)----- WHY DO YOU RETURN NULL
      fftw.execute(plan)
   
   
    if(backward) then
      output=torch.div(output,x:size(k))   
   end

  -- return torch.cat(ro,io,x:nDimension()+1)
      return output

end


--[[
-- BUGGY
function my_fft.my_fft_real(x,k)   
   assert(not tools.is_complex(x),'Signal is not real')
   assert(k<=x:nDimension(),'k is bigger than the dimension')
   local input=x
   
   --- Parameters of the fft PLAN
   local rank=1
   
   local n=torch.LongTensor({{input:size(k)}})
   local n_data=torch.data(n)
   local n_data_cast= ffi.cast('const int*',n_data)

   


      
   --input = input:contiguous() -- make sure input is contiguous
   local input_data = torch.data(input)
   local input_data_cast = ffi.cast('double*', input_data)
   
   local inembed_data_cast=n_data_cast--ffi.cast('const int*',0)
      
      
      
   local howmany=input:nElement()/input:size(k)     
   local istride=howmany--input:stride(k)
   local idist=input:nElement()/(input:stride(k)*input:size(k))
   idist=1


   local fs=torch.LongStorage(x:nDimension()+1)
   for l=1,input:nDimension() do
      fs[l]=input:size(l)
   end
   fs[input:nDimension()+1]=2
   
   local output = torch.Tensor(fs):typeAs(input):zero()
   
   local output_data = torch.data(output);
   local output_data_cast = ffi.cast(fftw_complex_cast, output_data)
   
   local oembed_data_cast=n_data_cast--ffi.cast('const int*',oembed_data)
      
   local ostride=istride
   local odist=idist
   
   
   local flags = fftw.ESTIMATE
   
   local plan  = fftw.plan_many_dft_r2c(rank,n_data_cast,howmany,input_data_cast,inembed_data_cast,
                                        istride,idist,output_data_cast,oembed_data_cast,ostride,odist,flags)   

-- fftw.execute(plan)
   
   
   local n_el=   torch.floor((input:size(k)-1)/2)
   local n_med= 2+torch.floor((input:size(k))/2)
   
   
   if(n_el>0) then
   output:narrow(k,n_med,n_el):indexCopy(k,torch.range(n_el,1,-1):long(),output:narrow(k,2,n_el))
   
      output:narrow(k,torch.ceil(input:size(k)/2)+1,torch.floor(input:size(k)/2)):narrow(output:nDimension(),2,1):mul(-1)
end
   
   fftw.destroy_plan(plan)
   
   return output
end

-- BUGGY
function my_fft.my_fft_complex(x,k,backward) 
   
   assert(tools.is_complex(x),'Signal is not complex')
   assert(k<x:nDimension(),'k is bigger than the dimension')  
   local input
   local flag=backward or false      
      input=x
   local rank=1
   
   local n=torch.LongTensor({{input:size(k)}})
   local n_data=torch.data(n)
   local n_data_cast= ffi.cast('const int*',n_data)
   
   
   local howmany=input:nElement()/(2*input:size(k))   
      
      input = input:contiguous() -- make sure input is contiguous
   local input_data = torch.data(input)
   local input_data_cast = ffi.cast(fftw_complex_cast, input_data)
   
   local inembed_data_cast=n_data_cast--ffi.cast('const int*',0)
   
   local idist=2*input:nElement()/(input:stride(k)*input:size(k)) 
      idist=x:size(k)
   local istride=x:stride(k)/2--input:nElement()/(input:stride(k)*input:size(k)) --x:stride(k)/2--x:stride(k)/2--1--input:stride(k)/2
      print('strides:')
print(x:stride(k)/2)
print(istride)   

      print('idist:')
   print(x:size(k))
   print(idist)
   print(howmany)
   
   local output = torch.Tensor(input:size(),input:stride()):zero()
   
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
      
--fftw.execute(plan)
  --  print(x:size())

  -- print(istride)
  -- print(idist)
  -- print(howmany)
  print((howmany-1)*idist+istride*(input:size(k)-1)+1)
print(x:nElement()/2)   
      

   fftw.destroy_plan(plan)
   
   if(flag) then
      output=torch.div(output,input:size(k))   
   end
   
   return output
end
--]]
return my_fft