local complex = require 'complex'
local conv_lib = require 'conv_lib'
local my_fft = require 'my_fft'
local filters_bank=require 'filters_bank'

local wavelet_transform={}

function wavelet_transform.WT(x,filters,no_low_pass)
   local res=x.res
   local out={}
   local res=x.res
   local ds
   local J=filters.J
   local A={}
   local V={}
   local j=x.j   
   local s_pad=filters.size[res+1]
   local mini_batch=x.mini_batch

   assert(res>=0,'Resolution is not positive')
   assert(res<J,'Resolution is bigger than J?')
   assert(x.j<J,'Scale is bigger than J?')
   ds=torch.max(torch.Tensor({torch.floor(J)-res, 0}))
      
   local buff = x.signal  
   local xf = my_fft.my_fft_complex(my_fft.my_fft_real(buff,1+mini_batch),2+mini_batch)

   A.signal = complex.realize(conv_lib.my_convolution_2d(xf,filters.phi.signal[res+1],ds,mini_batch))
 

   A.j = filters.phi.j
   A.res = res+ds
   
   if(not no_low_pass) then
      local k=1
      for i=1,#filters.psi do
         if(filters.psi[i].j >= j+1) then
            ds = torch.max(torch.Tensor({torch.floor(filters.psi[i].j)-res,0}))
            V[k] = {}
            V[k].signal = conv_lib.my_convolution_2d(xf,filters.psi[i].signal[res+1],ds,mini_batch)
            V[k].j = filters.psi[i].j
            V[k].theta = filters.psi[i].theta
            V[k].res = res+ds
            V[k].mini_batch=mini_batch
            k=k+1
         end         
      end
   end
   
   local out={}
   out.A = A
   out.V = V
   return out
end

return wavelet_transform