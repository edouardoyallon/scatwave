require 'sys'
require 'cutorch'
cufft=require 'cuda/wrapper_CUDA_fft'

x=torch.randn(256,256,2)
x=x:float()
xc=x:cuda()
y=torch.randn(32,32)
yc=y:cuda()
sys.tic()
cutorch.synchronize()
for i=1,20 do
   xf=cufft.my_fft_complex(xc,2)
end
print(xf[1][1][1])

print(sys.toc())  
fftw=require 'my_fft'   
   
  x=x:double()
   sys.tic()
for i=1,20 do
   xf2=fftw.my_fft_complex(fftw.my_fft_complex(x,2),1)
end
print(xf2[1][1][1])
print(sys.toc())  

--yf=cufft.my_fft_real(yc,2)
--print(xf[2][1])--]]--