require 'torch'
my_fft=require 'my_fft'

x=torch.randn(3,37,13,199)
--print(x:size())
--print(x:stride())
x=x:view(13,3,37,199)
--print(x:size())
--print(x:stride())
y=my_fft.my_fft_GURU_real(x,2)
--print(y)