local conv_lib = require 'conv_lib'
local complex = require 'complex'
local my_fft = require 'my_fft'
local filters_bank ={}

function filters_bank.display_littlehood_paley(f)
   local A=torch.Tensor(f.size):fill(0)
   
   for i=9,#f.psi do
      A=A+complex.abs_value(f.psi[i].signal[1])
   end
   local B=torch.Tensor(f.size):fill(0)
-- fftshift
   
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
   local sz=torch.Tensor(J+1,2)

   for res=0,J do
   local N_p=2^J*torch.ceil((N+2*J)/2^J)
   N_p=torch.max(torch.Tensor({{N_p,1}}))/2^res

   local M_p=2^J*torch.ceil((M+2*J)/2^J)/2^res
      M_p=torch.max(torch.Tensor({{M_p,1}}))

      sz[res+1][1]=N_p
      sz[res+1][2]=M_p
   end
   return sz
end


function filters_bank.morlet_filters_bank_2D(N,M,J)
   local a=torch.randn(N,M)
   --   a=a:double()
   local filters={}
   local i=1
   filters.size=get_padding_size(N,M,J,J) -- min padding should be a power of 2
   filters.psi={}
   filters.J=J
   local res_MAX=J
   for j=0,J-1 do
      for theta=1,8 do
         filters.psi[i]={}
         filters.psi[i].signal = {}
         filters.psi[i].signal[1]=morlet_2d(filters.size[1][1],filters.size[1][2],0.8*2^j,0.5,3/4*3.1415/2^j,theta*3.1415/8,0,1)
         filters.psi[i].signal[1]=my_fft.my_fft_complex(my_fft.my_fft_complex(filters.psi[i].signal[1],1),2)           
         
         for res=2,res_MAX do
            
            
               filters.psi[i].signal[res]=conv_lib.periodize_along_k(conv_lib.periodize_along_k(filters.psi[i].signal[1],1,res-1),2,res-1)
            
            
         end
         filters.psi[i].j=j
         filters.psi[i].theta=theta
         
         i=i+1
         
      end
   end
   
   filters.phi={}
   filters.phi.signal={}
      filters.phi.signal[1]=morlet_2d(filters.size[1][1],filters.size[1][2],0.8*2^(J-1),1,0,0,0,1)
      filters.phi.signal[1]=my_fft.my_fft_complex(my_fft.my_fft_complex(filters.phi.signal[1],1),2)   
   
   for res=2,res_MAX do

      filters.phi.signal[res]=conv_lib.periodize_along_k(conv_lib.periodize_along_k(filters.phi.signal[1],1,res-1),2,res-1)
      
   end
   filters.phi.j=J
   
   return filters 
end


function morlet_2d(M,N,sigma,slant,xi,theta,offset,fft_shift,no_DC)
   
   local wv=torch.Tensor(N,M,2)
   
   
   
   local R=torch.Tensor({{torch.cos(theta),-torch.sin(theta)},{torch.sin(theta),torch.cos(theta)}}) -- conversion to the axis..
   local R_inv=torch.Tensor({{torch.cos(theta),torch.sin(theta)},{-torch.sin(theta),torch.cos(theta)}})


   local g_modulus=torch.Tensor(M,N)
   local g_phase=torch.Tensor(M,N,2)
   local A=torch.Tensor({{1,0},{0,slant^2}})

   local i_,j_
--local tx=torch.Tensor({{,}})
   local    tx=torch.Tensor(1,2)
   local    x=torch.Tensor(2,1)
   local tmp=R*A*R_inv/(2*sigma^2)
   for i=1,M do
      for j=1,N do
         tx[1][1]=i-(M/2)+offset-1
         tx[1][2]=j-(N/2)+offset-1
         x[1]=i-(M/2)+offset-1
         x[2]=j-(N/2)+offset-1
         
--         tx:resize(1,2)
         if(fft_shift) then

            i_=(i-(M/2)-1)%M+1
            j_=(j-(N/2)-1)%N+1
         else
            i_=i
            j_=j
         end
         
         g_modulus[i_][j_]=torch.exp(-tx*tmp*x)--tmp*x)

         g_phase[i_][j_]=complex.unit_complex(x[1][1]*xi*torch.cos(theta)+x[2][1]*xi*torch.sin(theta))

         
         
      end
   end
   wv=complex.multiply_real_and_complex_tensor(g_phase,g_modulus)
   local K=torch.squeeze(torch.sum(torch.sum(wv,1),2))/(torch.squeeze(torch.sum(torch.sum(g_modulus,1),2)))

   local gab=wv-complex.multiply_complex_number_and_real_tensor(K,g_modulus)

     --       wv[i_][j_]=g_modulus*g_phase

   -- Now let's substract the average
   --   K

   norm_factor=1/(2*3.1415*sigma^2/slant)
   gab:mul(norm_factor)
   return gab
end


return filters_bank