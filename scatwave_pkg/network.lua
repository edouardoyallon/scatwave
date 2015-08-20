local complex = require 'complex'
local filters_bank = require 'filters_bank'
local wavelet_transform = require 'wavelet_transform'
local conv_lib = require 'conv_lib'
local network = torch.class('network')


function network:__init(M,N,J,dimension_mini_batch)
   --local ScatteringNetwork={J=5,M=32,N=32,dimension_mini_batch=1} -- J scale, M width, N height, K number of angles, dimension_mini_batch, the dimension after which the scattering will be batched. For instance, this means that you can batch a set of 128 color images stored in a tensor of size 128x3x256x256 by setting dimension_mini_batch=3
   
   self.M=M
   self.N=N
   self.J=J or 2
   self.type='torch.FloatTensor'
   self.myTensor=torch.FloatTensor
   self.dimension_mini_batch=dimension_mini_batch or 1
   self.fft=require 'wrapper_fft'   
   self.filters=filters_bank.morlet_filters_bank_2D(self.N,self.M,self.J,self)
end


function network:cuda()
   -- Should have a similar call to cuda() function in cunn   
   self.type='torch.CudaTensor'
   self.myTensor=torch.CudaTensor   
      
      -- First, we CUDArize the filters
      -- Phi   
   for l=1,#self.filters.phi.signal do
      self.filters.phi.signal[l]=self.filters.phi.signal[l]:cuda()
   end
   -- Psi
   for k=1,#self.filters.psi do
      for l=1,#self.filters.psi[k].signal do
         self.filters.psi[k].signal[l]=self.filters.psi[k].signal[l]:cuda()
      end
   end

   self.fft = require 'cuda/wrapper_CUDA_fft_nvidia'
end

function network:float()
   -- Should have a similar call to cuda() function in cunn   
   self.type='torch.FloatTensor'
   self.myTensor=torch.FloatTensor
   
   -- First, we deCUDArize the filters
   -- Phi   
   for l=1,#self.filters.phi.signal do
      self.filters.phi.signal[l]=self.filters.phi.signal[l]:float()
   end
   -- Psi
   for k=1,#self.filters.psi do
      for l=1,#self.filters.psi[k].signal do
         self.filters.psi[k].signal[l]=self.filters.psi[k].signal[l]:float()
      end
   end
   
   self.fft = require 'wrapper_fft'
end

function network:get_filters()
   return self.filters
end

function network:WT(image_input)
   local x={}
   x.signal=image_input
   x.res=0
   x.j=0
   return wavelet_transform.WT(x,self)
end

-- Here, we minimize the creation of memory to avoid using garbage collector
function network:fast_scat(image_input)
      assert(self.type==image_input:type(),'Not the correct type')
      local mini_batch=self.dimension_mini_batch-1
   
end

-- Usual scattering
function network:scat_with_wavelets(image_input)
   assert(self.type==image_input:type(),'Not the correct type')
   
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
   U[1][1].signal=conv_lib.pad_signal_along_k(conv_lib.pad_signal_along_k(image_input, s_pad[1], 1+mini_batch,self.myTensor), s_pad[2], 2+mini_batch,self.myTensor) 
      U[1][1].res=0
   U[1][1].j=-1000 -- encode the fact to compute wavelet of scale 0
      U[1][1].mini_batch=mini_batch
   

   out=wavelet_transform.WT(U[1][1],self)

   U[2]=complex.modulus_wise(out.V)
   
   
   S[1][1]=out.A
   local ds=self.J
   S[1][1].signal=conv_lib.unpad_signal_along_k(conv_lib.unpad_signal_along_k(S[1][1].signal,image_input:size(1+mini_batch),1+mini_batch,ds,self.myTensor),image_input:size(2+mini_batch),2+mini_batch,ds,self.myTensor)
   local k=1
   for i=1,#U[2] do

         out=wavelet_transform.WT(U[2][i],self)

      S[2][i]=out.A
      S[2][i].signal=conv_lib.unpad_signal_along_k(conv_lib.unpad_signal_along_k(S[2][i].signal,image_input:size(1+mini_batch),1+mini_batch,ds,self.myTensor),image_input:size(2+mini_batch),2+mini_batch,ds,self.myTensor)
      


      for l=1,#out.V do
         U[3][k]=out.V[l]

         U[3][k].signal=complex.abs_value(U[3][k].signal)
 
         k=k+1
      end
   end
   
   k=1
      
   for i=1,#U[3] do

      out=wavelet_transform.WT(U[3][i],self,1)

      S[3][i]=out.A
      
      S[3][i].signal=conv_lib.unpad_signal_along_k(conv_lib.unpad_signal_along_k(S[3][i].signal,image_input:size(1+mini_batch),1+mini_batch,ds,self.myTensor),image_input:size(2+mini_batch),2+mini_batch,ds,self.myTensor)
      -- k=k+1
   end
   

   return S
end
return network
