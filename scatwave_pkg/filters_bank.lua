local conv_lib = require 'conv_lib'
local complex = require 'complex'
--local my_fft = require 'my_fft'
local filters_bank ={}

function filters_bank.display_littlehood_paley(f)
   local A=torch.Tensor(torch.LongStorage({f.size[1][1],f.size[1][2]})):zero()
   for i=1,#f.psi do
      A=A+complex.abs_value(f.psi[i].signal[1])
   end
   local B=torch.Tensor(torch.LongStorage({f.size[1][1],f.size[1][2]})):fill(0)
  
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

function get_padding_size(N,M,max_ds,J)
   local sz=torch.LongTensor(J+1,2)

   for res=0,J do
   local N_p=2^J*torch.ceil((N+2*2^J)/2^J)
   N_p=torch.max(torch.Tensor({{N_p,1}}))/2^res

   local M_p=2^J*torch.ceil((M+2*2^J)/2^J)/2^res
      M_p=torch.max(torch.Tensor({{M_p,1}}))
      sz[res+1][1]=N_p
      sz[res+1][2]=M_p
   end
   return sz
end


function filters_bank.morlet_filters_bank_2D(N,M,J,scat)
   local filters={}
   local i=1
   local my_fft=scat.fft
   filters.size=get_padding_size(N,M,J,J) -- min padding should be a power of 2
   filters.psi={}
   filters.J=J
   local res_MAX=J
   for j=0,J-1 do
      for theta=1,8 do
         filters.psi[i]={}
         filters.psi[i].signal = {}
         filters.psi[i].signal[1] = morlet_2d(filters.size[1][1], filters.size[1][2], 0.8*2^j, 0.5, 3/4*3.1415/2^j, theta*3.1415/8, 0, 1,scat)
--            print(filters.psi[i].signal[1]:type())         
         filters.psi[i].signal[1] = my_fft.my_2D_fft_complex(filters.psi[i].signal[1], 1) --my_fft.my_fft_complex(my_fft.my_fft_complex(filters.psi[i].signal[1], 1), 2)           

         for res=2,res_MAX do
            filters.psi[i].signal[res]=conv_lib.periodize_along_k(conv_lib.periodize_along_k(filters.psi[i].signal[1], 1, res-1), 2, res-1)

         end

         filters.psi[i].j=j
         filters.psi[i].theta=theta
         i=i+1
      end
   end
   filters.phi={}
   filters.phi.signal={}
   filters.phi.signal[1]=gabor_2d(filters.size[1][1], filters.size[1][2], 0.8*2^(J-1), 1, 0, 0, 0, 1,scat)

   filters.phi.signal[1]=my_fft.my_2D_fft_complex(filters.phi.signal[1])--my_fft.my_fft_complex(my_fft.my_fft_complex(filters.phi.signal[1],1),2)    
      
   for res=2,res_MAX do
      filters.phi.signal[res]=conv_lib.periodize_along_k(conv_lib.periodize_along_k(filters.phi.signal[1],1,res-1),2,res-1)
   end
   
   filters.phi.j=J   

   return filters 
end


function meshgrid(x,y) -- identical to MATLAB
   local xx = torch.repeatTensor(x, y:size(1),1)
   local yy = torch.repeatTensor(y:view(-1,1), 1, x:size(1))
   return xx, yy
end


function gabor_2d(M,N,sigma,slant,xi,theta,offset,fft_shift,scat)  
   local wv=scat.myTensor(N,M,2)  
   local R=scat.myTensor({{torch.cos(theta),-torch.sin(theta)},{torch.sin(theta),torch.cos(theta)}}) -- conversion to the axis..
   local R_inv=scat.myTensor({{torch.cos(theta),torch.sin(theta)},{-torch.sin(theta),torch.cos(theta)}})
   local g_modulus=scat.myTensor(M,N)
   local g_phase=scat.myTensor(M,N,2)
   local A=scat.myTensor({{1,0},{0,slant^2}})
   local tmp=R*A*R_inv/(2*sigma^2)
   local x=torch.linspace(offset+1-(M/2)-1,offset+M-(M/2)-1,M)
   local y=torch.linspace(offset+1-(N/2)-1,offset+N-(N/2)-1,N)
   
   -- Shift the variable by half of their lenght
   if(fft_shift) then      
      local x_tmp=scat.myTensor(M)
      local y_tmp=scat.myTensor(N)
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

   wv = complex.multiply_real_and_complex_tensor(g_phase,g_modulus,scat)
   return wv
end


function morlet_2d(M,N,sigma,slant,xi,theta,offset,fft_shift,scat)
   local wv=gabor_2d(M,N,sigma,slant,xi,theta,offset,fft_shift,scat)  
   local g_modulus=complex.realize(gabor_2d(M,N,sigma,slant,0,theta,offset,fft_shift,scat))  
   local K=torch.squeeze(torch.sum(torch.sum(wv,1),2))/(torch.squeeze(torch.sum(torch.sum(g_modulus,1),2)))

   local mor=wv-complex.multiply_complex_number_and_real_tensor(K,g_modulus,scat)
   local    norm_factor=1/(2*3.1415*sigma^2/slant)
   mor:mul(norm_factor)
   return mor
end

return filters_bank