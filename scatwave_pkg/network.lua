local complex = require 'complex'
local filters_bank = require 'filters_bank'
local wavelet_transform = require 'wavelet_transform'
local conv_lib=require 'conv_lib'
local network = torch.class('network')


function network:__init(M,N,J,dimension_mini_batch)
   --local ScatteringNetwork={J=5,M=32,N=32,dimension_mini_batch=1} -- J scale, M width, N height, K number of angles, dimension_mini_batch, the dimension after which the scattering will be batched. For instance, this means that you can batch a set of 128 color images stored in a tensor of size 128x3x256x256 by setting dimension_mini_batch=3
   
   self.M=M
   self.N=N
   self.J=J or 2
   self.dimension_mini_batch=dimension_mini_batch or 1
   self.filters=filters_bank.morlet_filters_bank_2D(self.N,self.M,self.J)
   
end

function network:get_filters()
   return self.filters
end

function network:WT(image_input)
   local x={}
   x.signal=image_input
   x.res=0
   x.j=0
   return wavelet_transform.WT(x,self.filters)
end



function network:scat(image_input)
   local mini_batch=self.dimension_mini_batch-1
   local S={}
   local U={}
   S[1]={}
   S[2]={}
   S[3]={}

   U[1]={}
   U[2]={}
   U[3]={}
   
   
   

   U[1][1]={}
   local res=0
   local s_pad=self.filters.size[res+1]
   U[1][1].signal=conv_lib.pad_signal_along_k(conv_lib.pad_signal_along_k(image_input, s_pad[1], 1+mini_batch), s_pad[2], 2+mini_batch) 
   U[1][1].res=0
   U[1][1].j=-1000 -- encode the fact to compute wavelet of scale 0
   U[1][1].mini_batch=mini_batch
   
   out=wavelet_transform.WT(U[1][1],self.filters)
   U[2]=complex.modulus_wise(out.V)


   S[1][1]=out.A
   local k=1
   for i=1,#U[2] do

      out=wavelet_transform.WT(U[2][i],self.filters)
      S[2][i]=out.A
      for l=1,#out.V do

      U[3][k]=out.V[l]

         U[3][k].signal=complex.abs_value(U[3][k].signal)

      
         k=k+1
      end
   end
   

   
   k=1
   for i=1,#U[3] do
      out=wavelet_transform.WT(U[3][i],self.filters,1)
      S[3][i]=out.A
     -- k=k+1
   end
      

   return S
end
return network
