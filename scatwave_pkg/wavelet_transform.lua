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

   assert(res>=0,'Resolution is not positive')
   assert(res<J,'Resolution is bigger than J?')
   assert(x.j<J,'Scale is bigger than J?')
   ds=torch.max(torch.Tensor({torch.floor(J)-res, 0}))
      
   local buff = conv_lib.pad_signal_along_k(conv_lib.pad_signal_along_k(x.signal, s_pad[1], 1), s_pad[2], 2)   
   local xf = my_fft.my_fft_complex(my_fft.my_fft_real(buff,1),2)

   A.signal = complex.realize(conv_lib.my_convolution_2d(xf,filters.phi.signal[res+1],ds))
   A.signal = conv_lib.unpad_signal_along_k(conv_lib.unpad_signal_along_k(A.signal,x.signal:size(1),1,ds),x.signal:size(2),2,ds)
   A.j = filters.phi.j
   A.res = res+ds
   
   if(not no_low_pass) then
      local k=1
      for i=1,#filters.psi do
         if(filters.psi[i].j >= j+1) then
            ds = torch.max(torch.Tensor({torch.floor(filters.psi[i].j)-res,0}))
            V[k] = {}
            V[k].signal = conv_lib.my_convolution_2d(xf,filters.psi[i].signal[res+1],ds)
            V[k].signal = conv_lib.unpad_signal_along_k(conv_lib.unpad_signal_along_k(V[k].signal,x.signal:size(1),1,ds),x.signal:size(2),2,ds)
            V[k].j = filters.psi[i].j
            V[k].theta = filters.psi[i].theta
            V[k].res = res+ds
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