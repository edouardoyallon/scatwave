local complex = require 'scatwave.complex'
local conv_lib = require 'scatwave.conv_lib'

local filters_bank=require 'scatwave.filters_bank'

local wavelet_transform={}

function wavelet_transform.WT(x,scat,no_low_pass)
   local filters=scat.filters
   local my_fft=scat.fft
   local myTensor=scat.myTensor
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

   local xf = my_fft.my_2D_fft_real_batch(x.signal,1+mini_batch)

   A.signal = complex.realize(conv_lib.my_convolution_2d(xf,filters.phi.signal[res+1],ds,mini_batch,my_fft,myTensor))


   A.j = filters.phi.j
   A.res = res+ds

   if(not no_low_pass) then
      local k=1
      for i=1,#filters.psi do
         if(filters.psi[i].j >= j+1) then
            ds = torch.max(torch.Tensor({torch.floor(filters.psi[i].j)-res,0}))
            V[k] = {}
            V[k].signal = conv_lib.my_convolution_2d(xf,filters.psi[i].signal[res+1],ds,mini_batch,my_fft,myTensor)
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
