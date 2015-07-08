local complex=require 'complex'
local conv_lib=require 'conv_lib'
local filters_bank=require 'filters_bank'

local wavelet_transform={}

function wavelet_transform.WT(x,filters,no_low_pass)

   s1=x.signal:size()

   s2=filters.size


local s_pad=s1[1]/s2[1]
  
   
   local res=x.res
   local out={}
   local res=x.res
   local ds
   local J=filters.J
   local A={}
   local V={}
   local j=x.j
   assert(res>=0,'Resolution is not positive')
   assert(res<J,'Resolution is bigger than J?')
   --assert(x.j>=0,'Scale is not positive')
   assert(x.j<J,'Scale is bigger than J?')
   ds=torch.max(torch.FloatTensor({torch.floor(J)-res,0}))
   local xf=conv_lib.my_fft_complex(conv_lib.my_fft_real(conv_lib.pad_signal(x.signal,1),1),2)

   A.signal=complex.realize(conv_lib.my_convolution_2d(xf,filters.phi.signal[res+1],ds))
   A.j=filters.phi.j
   A.res=res+ds

   if(not no_low_pass) then
      local k=1
      for i=1,#filters.psi do

         if(filters.psi[i].j >= j+1) then
            ds=torch.max(torch.FloatTensor({torch.floor(filters.psi[i].j)-res,0}))
            V[k]={}
            V[k].signal=conv_lib.my_convolution_2d(xf,filters.psi[i].signal[res+1],ds)

            V[k].j=filters.psi[i].j

            V[k].res=res+ds
            k=k+1
         end

      end
   end
   local out={}
   out.A=A
   out.V=V
   return out
end
   
return wavelet_transform