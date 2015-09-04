--[[
     ScatWave implementation of Scattering Network
     Written by Edouard Oyallon
     Team DATA ENS
     Copyright 2015
]]

local conv_lib = require 'scatwave.conv_lib'
local complex = require 'scatwave.complex'

local filters_bank ={}

function filters_bank.display_littlehood_paley(f,r)
   r= r or 1
   local A=torch.FloatTensor(torch.LongStorage({f.size[r][1],f.size[r][2]})):zero()
   
   for i=1,#f.psi do
      if(#f.psi[i].signal>=r) then
         A=A+complex.abs_value(f.psi[i].signal[r])
      end
   end
   local B=torch.Tensor(torch.LongStorage({f.size[r][1],f.size[r][2]})):fill(0)
   
   -- fftshift to display... 
   for i=1,A:size(1) do
      for j=1,A:size(2) do
         i_=(i-(A:size(1)/2)-1)%A:size(1)+1
         j_=(j-(A:size(2)/2)-1)%A:size(2)+1
         B[i_][j_]=A[i][j]
      end
   end
   return B
end



function get_multires_size(s,max_res)
   local J=max_res   
   local sz={}
   
   for res=0,J do      
      local M_p=2^J*torch.ceil((s[s:size()]+2*2^J)/2^J)/2^res
      M_p=torch.max(torch.Tensor({{M_p,1}}))
      local N_p=2^J*torch.ceil((s[s:size()]+2*2^J)/2^J)
      N_p=torch.max(torch.Tensor({{N_p,1}}))/2^res
      local s_=torch.LongStorage(s:size())
      s_:copy(s)
      
      s_[s_:size()-1]=M_p
      s_[s_:size()]=N_p
      sz[res+1]=s_
   end
   
   return sz
end

function get_multires_stride(s,max_res)
   local J=max_res
   local sz={}
   
   for res=0,J do      
      local M_p=2^J*torch.ceil((s[s:size()-1]+2*2^J)/2^J)/2^res
      M_p=torch.max(torch.Tensor({{M_p,1}}))
      local N_p=2^J*torch.ceil((s[s:size()]+2*2^J)/2^J)
      N_p=torch.max(torch.Tensor({{N_p,1}}))/2^res
      local s_=torch.LongStorage(s:size())
      for l=1,s:size()-2 do
         s_[l]=0
      end
      
      s_[s_:size()-1]=N_p
      s_[s_:size()]=1
      sz[res+1]=s_
   end
   
   return sz
end


function reduced_freq_res(x,res,k)
   local s=x:stride()
   for l=1,#s do
      if not l==k then
         s[l]=0
      end
   end
   local mask = torch.FloatTensor(x:size(),s):fill(1) -- Tensor
      
   local z=mask:narrow(k,x:size(k)*2^(-res-1)+1,x:size(k)*(1-2^(-res))):fill(0)
   local y=torch.cmul(x,mask)
   return conv_lib.periodize_along_k(y,k,res,1)
end

function filters_bank.morlet_filters_bank_2D(U0_dim,J,fft)
   local filters={}
   local i=1
   
   
   size_multi_res=get_multires_size(U0_dim,J) -- min padding should be a power of 2
      filters.psi={}   
      filters.size_multi_res=size_multi_res
   
   stride_multi_res=get_multires_stride(U0_dim,J)


   for j=0,J-1 do
      for theta=1,8 do
         filters.psi[i]={}
         filters.psi[i].signal = {}
         
         local psi = morlet_2d(size_multi_res[1][U0_dim:size()-1], size_multi_res[1][U0_dim:size()], 0.8*2^j, 0.5, 3/4*3.1415/2^j, theta*3.1415/8, 0, 1)
         
         psi = complex.realize(fft.my_2D_fft_complex(psi))         
         filters.psi[i].signal[1]=torch.FloatTensor(psi:storage(),psi:storageOffset(),size_multi_res[1],stride_multi_res[1])
         
         for res=2,j+1 do--res_MAX do
            local tmp_psi=reduced_freq_res(reduced_freq_res(psi,res-1,1),res-1,2)
            filters.psi[i].signal[res]=torch.FloatTensor(tmp_psi:storage(),tmp_psi:storageOffset(),size_multi_res[res],stride_multi_res[res])
         end
         
         filters.psi[i].j=j
         filters.psi[i].theta=theta
         i=i+1
      end
   end
   filters.phi={}
   filters.phi.signal={}
   
   
   local phi=gabor_2d(size_multi_res[1][U0_dim:size()-1], size_multi_res[1][U0_dim:size()], 0.8*2^(J-1), 1, 0, 0, 0, 1)
   phi=complex.realize(fft.my_2D_fft_complex(phi))
   
   
   filters.phi.signal[1]=torch.FloatTensor(phi:storage(),phi:storageOffset(),size_multi_res[1],stride_multi_res[1])
   
   for res=2,J+1 do
      local tmp_phi=reduced_freq_res(reduced_freq_res(phi,res-1,1),res-1,2)
      filters.phi.signal[res]=torch.FloatTensor(tmp_phi:storage(),tmp_phi:storageOffset(),size_multi_res[res],stride_multi_res[res])--reduced_freq_res(reduced_freq_res(filters.phi.signal[1],res-1,U0_dim:size()-1),res-1,U0_dim:size())
   end
   
   filters.phi.j=J   
      
   return filters 
end


function meshgrid(x,y) -- identical to MATLAB
   local xx = torch.repeatTensor(x, y:size(1),1)
   local yy = torch.repeatTensor(y:view(-1,1), 1, x:size(1))
   return xx, yy
end


function gabor_2d(M,N,sigma,slant,xi,theta,offset,fft_shift) 
   
   local wv=torch.FloatTensor(N,M,2)  
   local R=torch.FloatTensor({{torch.cos(theta),-torch.sin(theta)},{torch.sin(theta),torch.cos(theta)}}) -- conversion to the axis..
   local R_inv=torch.FloatTensor({{torch.cos(theta),torch.sin(theta)},{-torch.sin(theta),torch.cos(theta)}})
   local g_modulus=torch.FloatTensor(M,N)
   local g_phase=torch.FloatTensor(M,N,2)
   local A=torch.FloatTensor({{1,0},{0,slant^2}})
   local tmp=R*A*R_inv/(2*sigma^2)
   local x=torch.linspace(offset+1-(M/2)-1,offset+M-(M/2)-1,M)
   local y=torch.linspace(offset+1-(N/2)-1,offset+N-(N/2)-1,N)
   local i_
      
      -- Shift the variable by half of their lenght
   if(fft_shift) then      
      local x_tmp=torch.FloatTensor(M)
      local y_tmp=torch.FloatTensor(N)
      for i=1,M do
         i_=(i-(M/2)-1)%M+1
         x_tmp[i_]=x[i]
      end
      for i=1,N do
         i_=(i-(N/2)-1)%N+1
         y_tmp[i_]=y[i]
      end
      x = x_tmp
      y = y_tmp
   end
   
   xx,yy = meshgrid(x,y)
   g_modulus = torch.exp(-(torch.cmul(xx,xx)*tmp[1][1]+torch.cmul(xx,yy)*tmp[2][1]+torch.cmul(yy,xx)*tmp[1][2]+torch.cmul(yy,yy)*tmp[2][2])) -- this is simply the result exp(-x^T*tmp*x)
      g_phase = xx*xi*torch.cos(theta)+yy*xi*torch.sin(theta)
   g_phase = complex.unit_complex(g_phase)
   
   wv = complex.multiply_real_and_complex_tensor(g_phase,g_modulus)
   local    norm_factor=1/(2*3.1415*sigma^2/slant)
   wv:mul(norm_factor)
   
   return wv
end


function morlet_2d(M,N,sigma,slant,xi,theta,offset,fft_shift)
   
   local wv=gabor_2d(M,N,sigma,slant,xi,theta,offset,fft_shift)
   local g_modulus=complex.realize(gabor_2d(M,N,sigma,slant,0,theta,offset,fft_shift))
   local K=torch.squeeze(torch.sum(torch.sum(wv,1),2))/(torch.squeeze(torch.sum(torch.sum(g_modulus,1),2)))
   
   local mor=wv-complex.multiply_complex_number_and_real_tensor(K,g_modulus)
   
   return mor
end

return filters_bank
