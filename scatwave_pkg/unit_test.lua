local unit_test_scatnet={}

require 'cutorch'

tester = torch.Tester()
local complex = require 'complex'
local my_fft = require 'wrapper_fft'
local my_fft_CUDA = require 'cuda/wrapper_CUDA_fft_nvidia'
local conv_lib = require 'conv_lib'
local filters_bank = require 'filters_bank'
local myTensor=torch.FloatTensor

local TOL=5e-4 -- surprisingly, the tolerance should 10^-5 (6 digits of precision) for float numbers
   
function unit_test_scatnet.complex()
   local alpha=torch.Tensor(1)
   alpha[1]=0.23
   local u_a = complex.unit_complex(alpha)
   
   tester:asserteq(u_a:nDimension(),2,'Unit complex number dimension isn\'t 2')
   tester:asserteq(u_a:size(2),2,'Not a complex')
      
   local norm_u_a=torch.sqrt(u_a[1][1]^2+u_a[1][2]^2)
   tester:asserteq(norm_u_a,1,string.sub('Not norm equal to 1 > ',norm_u_a))
   
   tester:asserteq(complex.abs_value(u_a)[1],1,'Abs value not equal to 1')
end


function unit_test_scatnet.my_fft()   
   local playing = torch.randn(18,27,4,4):float()
   playing:div(torch.max(playing))
   local playing2 = torch.randn(4,57,2):float()
   playing2:div(torch.max(playing2))
   
   -- We assert iFFT is working on both real FFT and complex one.
   local ff = my_fft.my_2D_fft_real_batch(playing,3)
   local iff = my_fft.my_2D_fft_complex_batch(ff,3,1)
   iff = complex.realize(iff)
   
   playing3=torch.randn(18,27,4,4,2):float():zero()
   playing3:narrow(5,1,1):copy(playing)
   local ff3 = my_fft.my_2D_fft_complex_batch(playing3,3)
   local ff2 = my_fft.my_2D_fft_complex(playing2)
   local iff2 = my_fft.my_2D_fft_complex(ff2,1)
   
 
--   print(playing[1][1][1][1])
--   print(playing4[1][1][1][1])
   local res = torch.squeeze(torch.sum(torch.sum(complex.abs_value(playing2-iff2),1),2))

   tester:assertlt(res,TOL,'complex IFFT and FFT not equal: ')
   tester:assertlt(torch.squeeze(torch.sum(torch.sum(torch.sum(torch.sum(torch.abs(iff-playing),1),2),3),4)),TOL,'batched IFFT and FFT not equal: ')
   tester:assertlt(torch.squeeze(torch.sum(torch.sum(torch.sum(torch.sum(complex.abs_value(ff3-ff),1),2),3),4)),TOL,'batched FFT real and complex FFT of a real signal not equal: ')
   
     -- Check that complex to real FFT is working...
   local playing4 = my_fft.my_2D_ifft_complex_to_real_batch(ff,3)
   local res3 = torch.squeeze(torch.sum(torch.sum(torch.sum(torch.sum(playing-playing4,1),2),3),4))
   
      tester:assertlt(res3,TOL,'complex to real IFFT and real not equal: ')

end

function unit_test_scatnet.my_fft_CUDA()   
   local playing_c = torch.randn(1,1,4,4):cuda()
   playing_c:div(torch.max(playing_c))
   local playing = playing_c:float()
   
   local playing2_c = torch.randn(1,1,27,4,2):cuda()
   playing2_c:div(torch.max(playing2_c))
   local playing2 = playing2_c:float()
   
   -- We assert iFFT is working on both real FFT and complex one.
   local ff = my_fft.my_2D_fft_real_batch(playing,3)
   local ff_c = my_fft_CUDA.my_2D_fft_real_batch(playing_c,3)
   
   local ff2_c = my_fft_CUDA.my_2D_fft_complex_batch(playing2_c,3)
   local ff2 = my_fft.my_2D_fft_complex_batch(playing2,3)
   
   ff_c=ff_c:float()
   ff2_c=ff2_c:float()
   local res=torch.squeeze(torch.sum(torch.sum(torch.sum(torch.sum(complex.abs_value(ff-ff_c),1),2),3),4))
   local res2=torch.squeeze(torch.sum(torch.sum(torch.sum(torch.sum(complex.abs_value(ff2_c-ff2_c),1),2),3),4))
   
   tester:assertlt(res,TOL,'real CUDA FFT and non CUDA are not equal')
   tester:assertlt(res2,TOL,'complex CUDA FFT are not equal')
end

function unit_test_scatnet.filters_bank()
   local N,M
   for k=1,10 do
      N=64+3*k
      M=65+3*5
      for J=1,5 do
         filters = filters_bank.morlet_filters_bank_2D(N,M,J,my_fft,myTensor)   
            -- Make sure the padding size is divisible by 2^J
         local si=filters.size
         tester:assertlt(si[1][2]%2^J+si[1][1]%2^J,TOL,'the padded size is not equal to 2^J')
         
         for s=1,#filters.psi do      
            for l=1,#filters.psi[s].signal do
               -- Make sure that all the filters are real
               local r = torch.squeeze(torch.sum(torch.sum(torch.squeeze(filters.psi[s].signal[l]:narrow(3,2,1)),1),2))
               tester:assertlt(r,TOL,'the filters are not real in Fourier') 
                  
                  -- Make sure the center frequency is (1,1) in computer coordinates
               local c=torch.squeeze(torch.sum(torch.abs(filters.psi[s].signal[l][1][1]),1))
               tester:assertlt(c,TOL,'the filters are not of mean 0...')
            end
         end
      end
   end
end
function unit_test_scatnet.wavelet()
   -- Check that the WT is indeed working   
end


function unit_test_scatnet.conv_lib()
   -- First we check that pad > unpad gives the identity
   for i=1,100 do
      local x=torch.randn(i,i):float()
      local z=conv_lib.pad_signal_along_k(conv_lib.pad_signal_along_k(x,i+torch.ceil(i/2),1,myTensor),i+torch.ceil(i/2),2,myTensor)
      local y=conv_lib.unpad_signal_along_k(conv_lib.unpad_signal_along_k(z,i,1,0,myTensor),i,2,0,myTensor)
      tester:asserteq(torch.squeeze(torch.sum(torch.sum(torch.abs(x-y),1),2)),0,'Pad > Unpading is not the identity')
   end
end

tester:add(unit_test_scatnet)
tester:run()

return unit_test_scatnet