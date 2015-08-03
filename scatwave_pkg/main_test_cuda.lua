require 'cutorch'
require 'gnuplot'
require 'image'
require 'env'
require 'sys'



x=image.lena()
--x=image.scale(x,256,256)

x=torch.randn(128,3,32,32)
x=x:float()
x=x:cuda()
-- Let's launch ScatWave!
ScatWave = require 'init'

t=0

sys.tic()
y=ScatWave.network.new(32,32,2,3) -- constructor to build a network
y:cuda()  -- make the network using CUDA optimizing it for x

-- Let's try to get the scattering coefficients
for i=1,10 do
   z=y:scat(x)
   if(i%2==0) then
      collectgarbage()   
   end
end

print(sys.toc()/10)

