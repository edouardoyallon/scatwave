local complex = require 'scatwave.complex'
local filters_bank = require 'scatwave.filters_bank'
local wavelet_transform = require 'scatwave.wavelet_transform'
local conv_lib = require 'scatwave.conv_lib'
local network = torch.class('network')


function network:__init(J,U0_dim)
   self.J=J or 3
   self.type='torch.FloatTensor'
   
   self.U0_dim=U0_dim or torch.LongTensor()

   self.fft=require 'scatwave.wrapper_fft'   
      
      self.filters=filters_bank.morlet_filters_bank_2D(U0_dim,self.J,self.fft)
   
   -- Allocate temporary variables
   local size_multi_res=self.filters.size_multi_res
   
   function allocate_multi_res(isComplex)
      x={}
      
      for r=1,size_multi_res:size(1) do
         
         local s
            if(isComplex) then
            s = tools.concatenateLongStorage(size_multi_res[r],torch.LongStorage({2}))
         else
            s=size_multi_res[r]
         end


         x[r]=torch.FloatTensor(size_multi_res[r]):fill(0)
      end
      return x
   end

   
   
   
   -- Input signal
   local size_r_2 = U0_dim
   size_r_2[U0_dim:nDimension()-1]=size_multi_res[1][U0_dim:nDimension()-1]
   self.U0_r_1 = torch.FloatTensor(U0_dim):fill(0) -- Auxiliary variable for padding because Torch is not efficient enough
      self.U0_r_2  = torch.FloatTensor(size_r_2):fill(0)
      self.U0_c = torch.FloatTensor(size_multi_res[r]):fill(0)
   
   -- First layer temporary variables
   self.U1_c = allocate_multi_res(1)
   self.U1_r = allocate_multi_res()

   -- Second layer temporary variables
   self.U2_c = allocate_multi_res(1)
   self.U2_r = allocate_multi_res()

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
   
   self.fft = require 'scatwave.cuda/wrapper_CUDA_fft_nvidia'
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

function network:allocate_inplace(mini_batch_dim)
   local filters=self.filters
   local myTensor=self.myTensor
   local filters_ip={}
   filters_ip.psi={}
   filters_ip.phi={}
   
  
   
   for i=1,#filters.psi do
      filters_ip.psi[i]={}
      filters_ip.psi[i].signal={}
      filters_ip.psi[i].j=filters.psi[i].j
      for r=1,#filters.psi[i].signal do
         local tmp=complex.realize(filters.psi[i].signal[r])
         local sizes=concatenateLongStorage(mini_batch_dim,tmp:size())
         local strides=torch.LongStorage(#sizes)
         for l=1,#sizes do
            if(l<=#mini_batch_dim) then
               strides[l]=0
            else
               strides[l]=tmp:stride(l-#mini_batch_dim)
            end
         end
         filters_ip.psi[i].signal[r]=myTensor(tmp:storage(),tmp:storageOffset(),sizes,strides)
      end
   end
   
   filters_ip.phi.signal={}
   for r=1,#filters.phi.signal do
      local tmp=complex.realize(filters.phi.signal[r])
      local sizes=concatenateLongStorage(mini_batch_dim,tmp:size())
      local strides=torch.LongStorage(#sizes)
      for l=1,#sizes do
         if(l<=#mini_batch_dim) then
            strides[l]=0
         else
            strides[l]=tmp:stride(l-#mini_batch_dim)
         end
      end
      filters_ip.phi.signal[r]=myTensor(tmp:storage(),tmp:storageOffset(),sizes,strides)
      
   end
   
   
   filters_ip.J=filters.J
   filters_ip.n_f=1+8*filters.J+8*8*filters.J*(filters.J-1)/2
   self.ip={}
   self.ip.filters=filters_ip
   self.ip.filters.size=filters.size
   
   self.ip.U1_c={}
   self.ip.U1_r={}
   self.ip.U2_c={}
   self.ip.U2_r={}
   self.ip.TMP_cuda_buggy={}   
   for r=1,filters.size:size(1) do
      local sz_r=concatenateLongStorage(mini_batch_dim,torch.LongStorage({filters.size[r][1],filters.size[r][2]}))
      local sz_c=concatenateLongStorage(mini_batch_dim,torch.LongStorage({filters.size[r][1],filters.size[r][2],2}))
      self.ip.U1_r[r]=myTensor(sz_r):fill(0)
      self.ip.TMP_cuda_buggy[r]=myTensor(sz_c):fill(0)
      self.ip.U2_r[r]=myTensor(sz_r):fill(0)
      self.ip.U1_c[r]=myTensor(sz_c):fill(0)
      self.ip.U2_c[r]=myTensor(sz_c):fill(0)
      
   end
   local J=filters.J
   
   f_size_along_x=1+torch.floor((self.M-1)/(2^J))
   f_size_along_y=1+torch.floor((self.N-1)/(2^J))
   
   self.ip.S=myTensor(concatenateLongStorage(mini_batch_dim,torch.LongStorage({filters_ip.n_f,f_size_along_x,f_size_along_y}))):fill(0)
   self.ip.x=myTensor(concatenateLongStorage(mini_batch_dim,torch.LongStorage({filters.size[1][1],filters.size[1][2]}))):fill(0)
   self.ip.x_=myTensor(concatenateLongStorage(mini_batch_dim,torch.LongStorage({self.M,filters.size[1][2]}))):fill(0)
   self.ip.xf=myTensor(concatenateLongStorage(mini_batch_dim,torch.LongStorage({filters.size[1][1],filters.size[1][2],2}))):fill(0)
   
end

-- CF nn.utils.recursiveType to convert float, tensor...


-- Here, we minimize the creation of memory to avoid using garbage collector
function network:scat(U0_r)
   assert(self.type==image_input:type(),'Not the correct type')
   -- Assert correct size also
   -- Assert contiguous
   
   -- Routines used for the computations
   local fft = self.fft  
   local filters = self.filters
   
   -- Minibatch variables (size, number of dim)
   local mini_batch_size = self.dimension_mini_batch
   local mini_batch_ndim = mini_batch_size:nElement()
   
   -- Input signal
   local U0_r_1 = self.U0_r_1 -- Auxiliary variable for padding because Torch is not efficient enough
   local U0_r_2 = self.U0_r_2     
   local U0_c = self.U0_c
   
   -- First layer temporary variables
   local U1_c = self.U1_c
   local U1_r = self.U1_r
   
   -- Second layer temporary variables
   local U2_c = self.U2_c
   local U2_r = self.U2_r
   
   -- Averaged scattering final variable
   local count_S_r = 1   
   local S_r = self.S_r
   local J = self.J
   
   
   -- Just because Torch7.0 is buggy -- reported @ https://github.com/torch/cutorch/issues/36#issuecomment-132765774
   local TMP= self.ip.TMP_cuda_buggy
   
   
   -- Pad the signal along the first, then second dimension. This operation can not be done jointly because Torch does not handle tensor copy in readonly mode
   conv_lib.pad_signal_along_k(U0_r, filters_ip.size[1][1], mini_batch_ndm,U0_r_1)
   conv_lib.pad_signal_along_k(U0_r_1, filters_ip.size[1][2], mini_batch_ndim+1,U0_r_2)      
      
      -- All the computations will be done with the padded signal. Only one unpadding will be applied on the output averaged signal.
   local function unpad_output_signal(ds)
      return  ds:narrow(mini_batch_ndim,2,S_r:size(mini_batch_ndim+1)):narrow(mini_batch_ndm+1,2,S_r:size(mini_batch+2))               
   end
   
   -- FFT of the input image
   -- fft.my_2D_fft_real_batch(U0_r_2,mini_batch_ndim,U0_c)    
   TMP[1]:narrow(TMP[1]:nDimension(),1,1):copy(U0_r_2)      
      fft.my_2D_fft_complex_batch(TMP[1],mini_batch_ndim,nil,U0_c)
   
   -- Compute the multiplication with xf and the LF, store it in U1_c[1]
   complex.multiply_complex_tensor_with_real_tensor_in_place(U0_c,filters.phi.signal[1],U1_c[1])
   
   
   -- Compute the complex to real iFFT of U1_c[1] and store it in U1_r[1]
   fft.my_2D_ifft_complex_to_real_batch(U1_c[1],mini_batch_ndim,U1_r[1])
   
   -- Store the downsample in S[k] where k is the corresponding position in the memory, k<-k+1   
   S_r:narrow(mini_batch_ndim,count_S_r,1):copy(unpad_output_signal(conv_lib.downsample_2D_inplace(U1_r[1],J,mini_batch_ndim)))
   count_S_r=count_S_r+1
   
   for j1=1,#filters.psi do
      -- Compute the multiplication with xf and the filters which is real in Fourier, finally store it in U1_c[1]
      local J1=filters_ip.psi[j1].j
      complex.multiply_complex_tensor_with_real_tensor_in_place(U0_c,filters.psi[j1].signal[1],U1_c[1])
      
      -- In this version, we do not periodize the signal 
      
      -- Compute the iFFT of U1_c[1], and store it in U1_c[1]      
      fft.my_2D_fft_complex_batch(U1_c[1],mini_batch,1,U1_c[1])
      
      -- We subsample it manually by changing its stride and store the subsampling in U1_c[j1]      
      U1_c[J1+1]:copy(conv_lib.downsample_2D_inplace(U1_c[1],J1,mini_batch_ndim))
      
      -- Compute the modulus and store it in U1_r[j1]
      complex.abs_value_inplace(U1_c[J1+1],U1_r[J1+1])
      
      -- Compute the Fourier transform and store it in U1_c[j1]
      --      fft.my_2D_fft_real_batch(U1_r[J1+1],mini_batch,U1_c[J1+1])
      TMP[J1+1]:narrow(TMP[J1+1]:nDimension(),1,1):copy(U1_r[J1+1])      
         fft.my_2D_fft_complex_batch(TMP[J1+1],mini_batch,nil,U1_c[J1+1])
      
      -- Compute the multiplication with U1_c[j1] and the LF, store it in U2_c[j1]
      complex.multiply_complex_tensor_with_real_tensor_in_place(U1_c[J1+1],filters_ip.phi.signal[J1+1],U2_c[J1+1])
      
      -- Compute the iFFT complex to real of U2_c[j1] and store it in U1_r[j1]
      wrapper_fft.my_2D_ifft_complex_to_real_batch(U2_c[J1+1],mini_batch,U1_r[J1+1])
      
      -- Store the downsample in S_r
      S_r:narrow(mini_batch_ndim,count_S_r,1):copy(unpad_output_signal(conv_lib.downsample_2D_inplace(U1_r[J1+1],J-J1,mini_batch_ndim)))
      count_S_r=count_S_r+1
      
      for j2=1,#filters_ip.psi do
         local J2=filters_ip.psi[j2].j         
         if (J2>J1) then
            -- Compute the multiplication with U1_c[j1] and the filters, and store it in U2_c[j1]
            complex.multiply_complex_tensor_with_real_tensor_in_place(U1_c[J1+1],filters_ip.psi[j2].signal[J1+1],U2_c[J1+1])
            
            -- Compute the iFFT of U2_c[j1], and store it in U2_c[j1]
            fft.my_2D_fft_complex_batch(U2_c[J1+1],mini_batch_ndim,1,U2_c[J1+1])         
               
               -- Subsample it and store it in U2_c[j2]
               U2_c[J2+1]:copy(conv_lib.downsample_2D_inplace(U2_c[J1+1],J2-J1,mini_batch_ndim))
            
            -- Compute the modulus and store it in U2_r[j2]
            complex.abs_value_inplace(U2_c[J2+1],U2_r[J2+1])
            
            -- Compute the Fourier transform of U2_r[j2] and store it in U2_c[j2]
            fft.my_2D_fft_real_batch(U2_r[J2+1],mini_batch_ndim,U2_c[J2+1])
            
            -- Compute the multiplication with U2_c[j2] and the LF, store it in U2_c[j2]    
            complex.multiply_complex_tensor_with_real_tensor_in_place(U2_c[J2+1],filters_ip.phi.signal[J2+1],U2_c[J2+1])
            
            -- Compute the complex to real iFFT of U2_c[j2] and store it in U2_r[j2]
            fft.my_2D_ifft_complex_to_real_batch(U2_c[J2+1],mini_batch,U2_r[J2+1])
            
            -- Store the downsample in S_r
            S_r:narrow(mini_batch_ndim,count_S_r,1):copy(unpad_output_signal(conv_lib.downsample_2D_inplace(U2_r[J2+1],J-J2,mini_batch_ndim)))
            count_S_r=count_S_r+1 
         end
      end
   end
   
   return S_r
end


function network:scat_inplace_DS(image_input)
   assert(self.type==image_input:type(),'Not the correct type')
   
   
   local mini_batch = self.dimension_mini_batch
   local x=self.ip.x
   local x_=self.ip.x_
   local filters_ip = self.ip.filters
   --conv_lib.pad_signal_along_k(image_input,,mini_batch,nil,x)
   
   conv_lib.pad_signal_along_k(image_input,filters_ip.size[1][1],mini_batch,nil,x_)
   conv_lib.pad_signal_along_k(x_,filters_ip.size[1][2],mini_batch+1,nil,x)
   
   local wrapper_fft=self.fft
   
   local myTensor=self.myTensor
   local xf=self.ip.xf
   local U1_c=self.ip.U1_c
   local U1_r=self.ip.U1_r
   local S=self.ip.S
   local U2_c=self.ip.U2_c
   local U2_r=self.ip.U2_r
   local ds
   local filters_ip = self.ip.filters
   local k=1
   local J=filters_ip.J
   local TMP= self.ip.TMP_cuda_buggy
   
   local decay_x=2--torch.floor((filters_ip.size[filters_ip.size:size(1)][1]*2^J-S:size(mini_batch+1))/2^(J+1))+1   
   local decay_y=2--torch.floor((filters_ip.size[filters_ip.size:size(1)][2]*2^J-S:size(mini_batch+2))/2^(J+1))+1         
      
      -- FFT of the input image
      --      wrapper_fft.my_2D_fft_real_batch(x,mini_batch,xf)
      TMP[1]:narrow(TMP[1]:nDimension(),1,1):copy(x)      
      wrapper_fft.my_2D_fft_complex_batch(TMP[1],mini_batch,nil,xf)
   
   -- Compute the multiplication with xf and the LF, store it in U1_c[1]
   complex.multiply_complex_tensor_with_real_tensor_in_place(xf,filters_ip.phi.signal[1],U1_c[1])
   
   -- Compute the complex to real iFFT of U1_c[1] and store it in U1_r[1]
   complex.periodize_in_place(U1_c[1],J,mini_batch,U1_c[J+1])
   
   -- Store the downsample in S[k] where k is the corresponding position in the memory, k<-k+1
   
   --   ds=conv_lib.downsample_2D_inplace(U1_r[1],J-1,mini_batch,myTensor)
   ds=wrapper_fft.my_2D_ifft_complex_to_real_batch(U1_c[J+1],mini_batch,U1_r[J+1])
   
   S:narrow(mini_batch,k,1):copy(ds:narrow(mini_batch,decay_x,S:size(mini_batch+1)):narrow(mini_batch+1,decay_y,S:size(mini_batch+2)))
   k=k+1
   
   for j1=1,#filters_ip.psi do
      -- Compute the multiplication with xf and the filters which is real in Fourier, finally store it in U1_c[1]
      local J1=filters_ip.psi[j1].j
      complex.multiply_complex_tensor_with_real_tensor_in_place(xf,filters_ip.psi[j1].signal[1],U1_c[1])
      
      -- Since cuFFT is fast, we do not periodize the signal      
      -- Compute the iFFT of U1_c[1], and store it in U1_c[1]      
      complex.periodize_in_place(U1_c[1],J1,mini_batch,U1_c[J1+1])
      wrapper_fft.my_2D_fft_complex_batch(U1_c[1],mini_batch,1,U1_c[1])
      -- We subsample it manually by changing its stride and store the subsampling in U1_c[j1]
      
      --      U1_c[J1+1]:copy(conv_lib.downsample_2D_inplace(U1_c[1],J1,mini_batch,myTensor))
      -- Compute the modulus and store it in U1_r[j1]
      
      complex.abs_value_inplace(U1_c[J1+1],U1_r[J1+1])
      
      -- Compute the Fourier transform and store it in U1_c[j1]
      --      wrapper_fft.my_2D_fft_real_batch(U1_r[J1+1],mini_batch,U1_c[J1+1])
      TMP[J1+1]:narrow(TMP[J1+1]:nDimension(),1,1):copy(U1_r[J1+1])      
         wrapper_fft.my_2D_fft_complex_batch(TMP[J1+1],mini_batch,nil,U1_c[J1+1])
      
      -- Compute the multiplication with U1_c[j1] and the LF, store it in U2_c[j1]
      complex.multiply_complex_tensor_with_real_tensor_in_place(U1_c[J1+1],filters_ip.phi.signal[J1+1],U2_c[J1+1])
      
      -- Compute the iFFT complex to real of U2_c[j1] and store it in U1_r[j1]
      
      complex.periodize_in_place(U2_c[J1+1],J-J1,mini_batch,U2_c[J+1])
      
      
      -- Store the downsample in S[k] where k is the corresponding position in the memory, k<-k+1
      ds=wrapper_fft.my_2D_ifft_complex_to_real_batch(U2_c[J+1],mini_batch,U1_r[J+1])
      
      S:narrow(mini_batch,k,1):copy(ds:narrow(mini_batch,decay_x,S:size(mini_batch+1)):narrow(mini_batch+1,decay_y,S:size(mini_batch+2)))
      k=k+1
      
      for j2=1,#filters_ip.psi do
         
         -- for j2
         if (filters_ip.psi[j2].j>filters_ip.psi[j1].j) then
            local J2=filters_ip.psi[j2].j
            -- Compute the multiplication with U1_c[j1] and the filters, and store it in U2_c[j1]
            complex.multiply_complex_tensor_with_real_tensor_in_place(U1_c[J1+1],filters_ip.psi[j2].signal[J1+1],U2_c[J1+1])
            
            complex.periodize_in_place(U2_c[J1+1],J2-J1,mini_batch,U2_c[J2+1])
            --U2_c[J2+1]:copy(conv_lib.downsample_2D_inplace(U2_c[J1+1],J2-J1,mini_batch,myTensor))
            -- Compute the iFFT of U2_c[j1], and store it in U2_c[j1]
            wrapper_fft.my_2D_fft_complex_batch(U2_c[J2+1],mini_batch,1,U2_c[J2+1])         
               
               -- Subsample it and store it in U2_c[j2]
               
               
               -- Compute the modulus and store it in U2_r[j2]
               complex.abs_value_inplace(U2_c[J2+1],U2_r[J2+1])
            
            -- Compute the Fourier transform of U2_r[j2] and store it in U2_c[j2]
            wrapper_fft.my_2D_fft_real_batch(U2_r[J2+1],mini_batch,U2_c[J2+1])
            
            -- Compute the multiplication with U2_c[j2] and the LF, store it in U2_c[j2]    
            
            -- print(U2_c[J2+1]:size())
            -- print(filters_ip.phi.signal[J2+1]:size())
            complex.multiply_complex_tensor_with_real_tensor_in_place(U2_c[J2+1],filters_ip.phi.signal[J2+1],U2_c[J2+1])
            
            -- Compute the complex to real iFFT of U2_c[j2] and store it in U2_r[j2]
            complex.periodize_in_place(U2_c[J2+1],J-J2,mini_batch,U2_c[J+1])          
               ds=wrapper_fft.my_2D_ifft_complex_to_real_batch(U2_c[J+1],mini_batch,U2_r[J+1])
            
            -- Store the downsample in S[k] where k is the corresponding position in the memory, k<-k+1
            --            ds=conv_lib.downsample_2D_inplace(U2_r[J2+1],J-J2-1,mini_batch,myTensor)
            S:narrow(mini_batch,k,1):copy(ds:narrow(mini_batch,decay_x,S:size(mini_batch+1)):narrow(mini_batch+1,decay_y,S:size(mini_batch+2)))
            k=k+1
            
         end
      end
   end
   
   --
   
   
   
   return S
end

return network
