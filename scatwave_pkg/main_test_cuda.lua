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
--   y:float()
y:cuda(x)  -- make the network using CUDA optimizing it for x
   
-- for images of size
print(sys.toc())   -- 32 by 32 invariant up to 2^2



-- Let's try to get the scattering coefficients
--   sys.tic()
for i=1,2 do

   z=y:scat(x)
--if(i%5==0) then
      collectgarbage()   
--end

end
   cutorch.synchronize()

--   print(sys.toc()/10)


-- Printing
-- print(z)

