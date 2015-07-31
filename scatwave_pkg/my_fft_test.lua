require 'torch'
my_fft=require 'my_fft'

x=torch.randn(4,4,2)
--print(x:size())
--print(x:stride())
--x=x:view(4,8,2)
--print(x:size())
--print(x:stride())
y=my_fft.my_fft_complex(x,2)

z=my_fft.my_fft_complex(y,2,1)
