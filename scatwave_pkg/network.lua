local complex = require 'complex'
local filters_bank = require 'filters_bank'
local wavelet_transform = require 'wavelet_transform'

local network = torch.class('network')


function network:__init(M,N,J)
   --local ScatteringNetwork={J=5,M=32,N=32} -- J scale, M width, N height, K number of angles
   
   --setmetatable(self.__index,ScatteringNetwork)
   self.M=M
   self.N=N
   self.J=J or 2
   self.filters=filters_bank.morlet_filters_bank_2D(self.N,self.M,self.J)
   
end

function network:get_filters()
   return self.filters
end

function network:WT(image_input)
   --local x=complex.unit_complex(10)
   local x={}
   x.signal=image_input
   x.res=0
   x.j=0
   return wavelet_transform.WT(x,self.filters)
end



function network:scat(image_input)
   --local x=complex.unit_complex(10)
   local x={}
   local S={}
   local U={}
   S[1]={}
   S[2]={}
   S[3]={}

   U[1]={}
   U[2]={}
   U[3]={}

   U[1][1]={}
   U[1][1].signal=image_input
   U[1][1].res=0
   U[1][1].j=-1000 -- encode the fact to compute wavelet of scale 0
   
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
