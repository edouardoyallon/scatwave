local unit_test_scatnet={}

require 'torch'
tester = torch.Tester()
local complex = require 'complex'
local my_fft = require 'my_fft'
local conv_lib = require 'conv_lib'

function unit_test_scatnet.complex()
   local playing = torch.Tensor(4,5,7,2)
   for i = 1,4 do
      for j = 1,5 do
         for k = 1, 7 do
            playing[i][j][k][1]=i*j*k
            playing[i][j][k][1]=torch.sqrt(i*j*k)
         end
      end
   end

   -- unit_complex
   local alpha=torch.Tensor(1)
   alpha[1]=0.23
   local u_a = complex.unit_complex(alpha)
   tester:asserteq(u_a:nDimension(),2,'Unit complex number dimension isn\'t 2')
   tester:asserteq(u_a:size(2),2,'Not a complex')
   
   local norm_u_a=torch.sqrt(u_a[1][1]^2+u_a[1][2]^2)
   tester:asserteq(norm_u_a,1,'Not norm equal to 1')
   
   -- complex.abs_value
   tester:asserteq(complex.abs_value(u_a)[1],1,'Abs value not equal to 1')
end


function unit_test_scatnet.my_fft()   
   local playing = torch.randn(4,4,18,29)
   local playing2 = torch.randn(4,5,18,29,2)
   
   -- We assert iFFT is working on both real FFT and complex one.
   for k=1,4 do
      local ff = my_fft.my_fft_real(playing,k)
      local iff = my_fft.my_fft_complex(ff,k,1)
      iff = complex.realize(iff)

      local ff2 = my_fft.my_fft_complex(playing2,k)
      local iff2 = my_fft.my_fft_complex(ff2,k,1)
      tester:assertlt(torch.squeeze(torch.sum(torch.sum(torch.sum(torch.sum(complex.abs_value(iff2-playing2),1),2),3),4)),10^-8,'IFFT and FFT not equal: ')
      tester:assertlt(torch.squeeze(torch.sum(torch.sum(torch.sum(torch.sum(torch.abs(iff-playing),1),2),3),4)),10^-8,'IFFT and FFT not equal: ')
   end
end


function unit_test_scatnet.conv_lib()
   -- First we check that pad > unpad gives the identity
   for i=1,100 do
      local x=torch.randn(i,i)
      local z=conv_lib.pad_signal_along_k(conv_lib.pad_signal_along_k(x,i+torch.ceil(i/2),1),i+torch.ceil(i/2),2)
      local y=conv_lib.unpad_signal_along_k(conv_lib.unpad_signal_along_k(z,i,1,0),i,2,0)

   tester:asserteq(torch.squeeze(torch.sum(torch.sum(torch.abs(x-y),1),2)),0,'Pad > Unpading is not the identity')
   end
   
   -- Then we check that the down sampling & periodization is working with resolution downsampling...
   
    for i=16,100 do
      local x=torch.linspace(1,i,i)
      for res=1,4 do
         local x_stamp=torch.randn(i*2^res)
         local ts=1
         for l=1,i do
            for r=1,2^res do
               x_stamp[ts] = x[l]
               ts = ts+1
            end
         end
         
         
         local pad_size=x_stamp:nElement()+2*2^res
         


         local z=conv_lib.pad_signal_along_k(x_stamp,pad_size,1)
         
         
         z=my_fft.my_fft_real(z,1)
         z=conv_lib.periodize_along_k(z,1,res)

         z=my_fft.my_fft_complex(z,1,1)
   
         z=complex.realize(z)

      local y=conv_lib.unpad_signal_along_k(z,i*2^res,1,res)
     


         tester:assertlt(torch.squeeze(torch.sum(torch.abs(x-y),1)),10^-10,'Issue')
         
      end
   end

   
end

tester:add(unit_test_scatnet)
tester:run()

return unit_test_scatnet
