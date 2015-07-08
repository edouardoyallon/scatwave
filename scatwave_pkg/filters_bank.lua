local conv_lib = require 'conv_lib'
local complex = require 'complex'
local filters_bank ={}



function filters_bank.morlet_filters_bank_2D(N,M,J)
   local a=torch.randn(N,M)
--   a=a:double()
   local filters={}
   local i=1
   filters.size=conv_lib.pad_signal(a):size()
   filters.psi={}
   filters.J=J
   local res_MAX=J
for j=0,J-1 do
      for theta=1,8 do
         filters.psi[i]={}
         filters.psi[i].signal = {}
         for res=1,res_MAX do
            
            filters.psi[i].signal[res]=morlet_2d(N,M,0.8*2^j,0.8*2^j,0.5,theta*3.1415/8,0,1)
          filters.psi[i].signal[res]=conv_lib.my_fft_complex(conv_lib.my_fft_complex(filters.psi[i].signal[res],1),2)  
         filters.psi[i].signal[res]=conv_lib.periodize_along_k(conv_lib.periodize_along_k(filters.psi[i].signal[res],1,res-1),2,res-1)
            

         end
         filters.psi[i].j=j
         filters.psi[i].theta=theta

i=i+1

end
   end
   
   filters.phi={}
   filters.phi.signal={}
   for res=1,res_MAX do
      filters.phi.signal[res]=morlet_2d(N,M,2^J,0.8,0,0,0,1)
      filters.phi.signal[res]=conv_lib.my_fft_complex(conv_lib.my_fft_complex(filters.phi.signal[res],1),2)
      filters.phi.signal[res]=conv_lib.periodize_along_k(conv_lib.periodize_along_k(filters.phi.signal[res],1,res-1),2,res-1)

end
   filters.phi.j=J

  return filters 
end


function morlet_2d(N,M,sigma,slant,xi,theta,offset,fft_shift)
   
   local wv=torch.Tensor(N,M,2)



   local R=torch.Tensor({{torch.cos(theta),torch.sin(theta)},{-torch.sin(theta),torch.cos(theta)}})
   --R=R:float()
   
   local R_inv=torch.Tensor({{torch.cos(theta),-torch.sin(theta)},{torch.sin(theta),torch.cos(theta)}})
   local x
   local tx
   local g_modulus
  local g_phase
    local A=torch.Tensor({{1,0},{0,slant^2}})
   for i=1,N do
      for j=1,N do
        
         tx=torch.Tensor({i-N/2+offset,j-N/2+offset})
          x=torch.Tensor({{i-N/2+offset},{j-N/2+offset}})
         tx:resize(1,2)
         if(fft_shift) then
            i_=(i+N/2-1)%N+1
            j_=(j+M/2-1)%M+1
         else
            i_=i
            j_=j
            end

         g_modulus=torch.exp(-tx*R_inv*A*R*x/sigma^2)

         g_phase=complex.unit_complex(x[1][1]*xi*torch.cos(theta)+x[2][1]*xi*torch.sin(theta))
         wv[i_][j_]=g_modulus*g_phase
         
         
      end
   end
   
   return wv
end


return filters_bank